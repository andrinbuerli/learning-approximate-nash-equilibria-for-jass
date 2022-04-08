import logging
from typing import Tuple

import numpy as np
from jass.game.const import next_player, TRUMP_FULL_OFFSET, TRUMP_FULL_P, team, PUSH_ALT, PUSH
from jasscpp import GameStateCpp, GameObservationCpp, RuleSchieberCpp, GameSimCpp

from lib.cfr.game_util import deal_random_hand, copy_state
np.seterr(all='raise')

class OOS:

    def __init__(
            self,
            delta: float,
            epsilon: float,
            gamma: float,
            action_space: int,
            players: int,
            chance_sampling: bool,
            iterations_per_chance_sample: int,
            log:bool = False):
        self.iterations_per_chance_sample = iterations_per_chance_sample
        self.chance_sampling = chance_sampling
        self.log = log
        self.players = players
        self.gamma = gamma
        self.action_space = action_space
        self.epsilon = epsilon
        self.delta = delta
        self.information_sets = {}  # infostate keys -> [cum. regrets, avg strategy, imm. regrets, valid_actions]
        self.rule = RuleSchieberCpp()

        if self.log:
            self.immediate_regrets = []

    def reset(self):
        self.immediate_regrets = []
        self.information_sets = {}

    def get_infostate_key(self, h: GameStateCpp):
        return f"{h.trump}:{h.forehand}:{np.flatnonzero(h.hands[h.player])}:{h.tricks[h.current_trick]}"

    def get_infostate_key_from_obs(self, m: Tuple[GameStateCpp, GameObservationCpp]):
        if isinstance(m, GameObservationCpp):
            return f"{m.trump}:{m.forehand}:{np.flatnonzero(m.hand)}:{m.tricks[m.current_trick]}"
        else:
            return self.get_infostate_key(m)

    def get_average_stragety(self, informationset_key: str):
        _, avg_strategy, _, valid_actions = self.information_sets[informationset_key]
        strategy_sum = avg_strategy.sum()

        if strategy_sum > 0:
            return avg_strategy / strategy_sum
        else:
            return valid_actions / valid_actions.sum()

    def run_iterations(self, m: Tuple[GameStateCpp, GameObservationCpp], iterations: int, targeted_mode_init: bool = None):

        if self.chance_sampling:
            iterations = iterations // self.iterations_per_chance_sample

        for j in range(iterations):
            if targeted_mode_init is None:
                targeted_mode = bool(np.random.choice([1, 0], p=[self.delta, 1 - self.delta]))
            else:
                targeted_mode = targeted_mode_init

            if targeted_mode:
                p_bar_h = 1
                known_hands = self.get_known_hands(m)
                hands_targeted, prob_targeted = deal_random_hand(known_hands=known_hands)
                p_bar_h *= prob_targeted

                game = GameSimCpp()
                game.state.hands = hands_targeted
                game.state.dealer = m.dealer
                game.state.player = next_player[m.dealer]

                if m.forehand > -1:
                    if m.forehand == 0:
                        a = TRUMP_FULL_P
                        key = self.get_infostate_key(game.state)
                        if key in self.information_sets:
                            p_a = self.get_average_stragety(key)[a]
                        else:
                            p_a = 1 / game.get_valid_actions().sum()
                        game.perform_action_full(a)
                        p_bar_h *= p_a

                    a = m.trump + TRUMP_FULL_OFFSET
                    key = self.get_infostate_key(game.state)
                    if key in self.information_sets:
                        p_a = self.get_average_stragety(key)[a]
                    else:
                        p_a = 1 / game.get_valid_actions().sum()
                    game.perform_action_full(a)
                    p_bar_h *= p_a

                for c in m.tricks.reshape(-1):
                    if c == -1:
                        break

                    key = self.get_infostate_key(game.state)
                    if key in self.information_sets:
                        p_a = self.get_average_stragety(key)[c]
                    else:
                        p_a = 1 / game.get_valid_actions().sum()

                    game.perform_action_full(c)
                    p_bar_h *= p_a

                w_T = (1 - self.delta) + self.delta * p_bar_h
            else:
                w_T = 1


            if self.chance_sampling:
                hands, prob_targeted, prob_untargeted = self.sample_chance_outcome(m, targeted_mode)

                for _ in range(self.iterations_per_chance_sample):
                    for i in range(self.players):
                        state = GameStateCpp()
                        state.dealer = m.dealer
                        state.player = next_player[m.dealer]
                        state.hands = hands
                        self.iterate(
                            m, state,
                            [1 for _ in range(self.players)], prob_untargeted * 1,
                                                              prob_targeted * w_T, prob_untargeted * w_T,
                            i, targeted_mode)
            else:
                for i in range(self.players):
                    state = GameStateCpp()
                    state.dealer = m.dealer
                    state.player = next_player[m.dealer]
                    self.iterate(m, state, [1 for _ in range(self.players)], 1, w_T, w_T, i, targeted_mode)

            if self.log:
                #imregret = np.nanmean([r.max() for key, (_, _, r) in self.infostates.items()])
                key = self.get_infostate_key_from_obs(m)
                if isinstance(m, GameObservationCpp):
                    valid_actions = np.flatnonzero(self.rule.get_full_valid_actions_from_obs(m))
                else:
                    valid_actions = np.flatnonzero(self.rule.get_full_valid_actions_from_state(m))

                if key in self.information_sets:
                    _, _, r, _ = self.information_sets[key]
                    imregret = r[valid_actions]
                else:
                    imregret = [-1 for _ in valid_actions]

                self.immediate_regrets.append(imregret)
                logging.info(f"Touched infosets: {len(self.information_sets)}, cards played: {m.nr_played_cards}, Average Regret: {imregret}")

    def sample_chance_outcome(self, m, targeted_mode):
        known_hands = self.get_known_hands(m)
        hands_targeted, prob_targeted = deal_random_hand(known_hands=known_hands)
        if isinstance(m, GameObservationCpp):
            known_hands = [hand if i == m.player else [] for i, hand in enumerate(known_hands)]
        hands_untargeted, prob_untargeted = deal_random_hand(known_hands=known_hands)
        if targeted_mode:
            hands = hands_targeted
        else:
            hands = hands_untargeted
        game = GameSimCpp()
        game.state.hands = hands_targeted
        game.state.dealer = m.dealer
        game.state.player = next_player[m.dealer]
        if m.forehand > -1:
            if m.forehand == 0:
                a = TRUMP_FULL_P
                key = self.get_infostate_key(game.state)
                if key not in self.information_sets:
                    self.add_information_set(key, game.get_valid_actions())
                game.perform_action_full(a)

            a = m.trump + TRUMP_FULL_OFFSET
            key = self.get_infostate_key(game.state)
            if key not in self.information_sets:
                self.add_information_set(key, game.get_valid_actions())
            game.perform_action_full(a)
        for c in m.tricks.reshape(-1):
            if c == -1:
                break

            key = self.get_infostate_key(game.state)
            if key not in self.information_sets:
                self.add_information_set(key, game.get_valid_actions())

            game.perform_action_full(c)
        assert hands.sum() == 36
        return hands, prob_targeted, prob_untargeted

    def iterate(
            self,
            m: Tuple[GameStateCpp, GameObservationCpp],
            h: GameStateCpp,
            pi_i: [float],
            pi_o: float,
            s_1: float,
            s_2: float,
            i: int,
            targeted_mode: bool,
            game = GameSimCpp()):
        """

        :param m: known history in ongoing game
        :param h: history in game sample
        :param pi_i: strategy reach probability for update player
        :param pi_o: strategy reach probability for opponent players
        :param s_1: overall probability that current sample is generated in targeted mode
        :param s_2: overall probability that current sample is generated in untargeted mode
        :param i: update player
        :param targeted_mode: update mode
        :return:
        """

        assert s_1 <= 1, f"invalid value for s_1: {s_1}"
        assert s_2 <= 1, f"invalid value for s_2: {s_2}"

        is_terminal_history = h.nr_played_cards == 36
        is_chance_node = h.hands.min() == -1
        if is_terminal_history:
            return 1, self.delta * s_1 + (1 - self.delta) * s_2, h.points
        elif is_chance_node:
            known_hands = self.get_known_hands(m)
            hands_targeted, prob_targeted = deal_random_hand(known_hands=known_hands)
            if isinstance(m, GameObservationCpp):
                known_hands = [hand if i == m.player else [] for i, hand in enumerate(known_hands)]
            hands_untargeted, prob_untargeted = deal_random_hand(known_hands=known_hands)

            if targeted_mode:
                h.hands = hands_targeted
            else:
                h.hands = hands_untargeted

            assert h.hands.sum() == 36

            return self.iterate(
                m, h,
                pi_i, prob_untargeted * pi_o,
                prob_targeted * s_1, prob_untargeted * s_2,
                i, targeted_mode)

        infoset_key = self.get_infostate_key(h)

        if infoset_key in self.information_sets:
            regret, avg_strategy, imm_regrets, valid_actions = self.information_sets[infoset_key]
            valid_actions_list = np.flatnonzero(valid_actions)
            assert (valid_actions == self.rule.get_full_valid_actions_from_state(h)).all()
            current_strategy = self.regret_matching(regret, valid_actions_list)
            if h.player == i:
                current_strategy = self.add_exploration(current_strategy, valid_actions_list)
        else:
            valid_actions = self.rule.get_full_valid_actions_from_state(h)
            valid_actions_list = np.flatnonzero(valid_actions)
            current_strategy = valid_actions / valid_actions.sum()

        a, s_1_prime, s_2_prime = self.sample(h, m, current_strategy, valid_actions_list, s_1, s_2, targeted_mode)

        # assert a in valid_actions_list, f"{a} not in {valid_actions}, h: {h}, m: {m}"

        if infoset_key not in self.information_sets:
            avg_strategy, imm_regrets, regret = self.add_information_set(infoset_key, valid_actions)
            x, l, u = self.playout(h, a, (self.delta * s_1 + (1 - self.delta) * s_2) / valid_actions.sum())
        else:
            pi_i_prime =  [current_strategy[a] * pi_ii if player == h.player else pi_ii for player, pi_ii in enumerate(pi_i)]
            pi_o_prime = current_strategy[a] * pi_o if h.player != i else pi_o

            h_tmp = copy_state(h)
            game.state = h
            game.perform_action_full(a)
            h_prime = game.state
            h = h_tmp
            x, l, u = self.iterate(m, h_prime, pi_i_prime, pi_o_prime, s_1_prime, s_2_prime, i, targeted_mode)

        c = x
        x = x * current_strategy[a]

        for v_a in valid_actions_list:
            if h.player == i:
                u_i = self.get_utility_for(u, i)
                W = u_i * pi_o / l
                if v_a == a:
                    imm_regrets[v_a] = (c - x) * W
                else:
                    imm_regrets[v_a] = - x * W

                regret[v_a] = regret[v_a] + imm_regrets[v_a]
            else:
                # TODO: FIX PROBLEMS WITH VALUES SMALLER THAN computable with float32 -> RUNTIME WARNINGS
                pi_i_I = (self.delta * s_1 + (1 - self.delta) * s_2)
                avg_strategy[v_a] = avg_strategy[v_a] \
                                    + ((pi_i[h.player] * current_strategy[v_a]) / pi_i_I)

        self.information_sets[infoset_key] = (regret, avg_strategy, imm_regrets, valid_actions)

        return x, l, u

    def add_information_set(self, informationset_key, valid_actions):
        regret = np.zeros(self.action_space)
        avg_strategy = np.zeros(self.action_space)
        imm_regrets = np.zeros(self.action_space)
        self.information_sets[informationset_key] = (
            regret, avg_strategy, imm_regrets, valid_actions
        )
        return avg_strategy, imm_regrets, regret

    def get_known_hands(self, m):
        if isinstance(m, GameStateCpp):
            known_hands = [np.flatnonzero(x).tolist() for x in m.hands]
        else:
            known_hands = [[] for _ in range(4)]
            known_hands[m.player] = np.flatnonzero(m.hand).tolist()
        for j, trick in enumerate(m.tricks):
            card_player = m.trick_first_player[j]
            for card in trick:
                if card == -1:
                    break
                known_hands[card_player].append(card)
                card_player = next_player[card_player]
        return known_hands

    def sample(
            self,
            h: GameStateCpp,
            m: GameObservationCpp,
            avg_strategy: np.array,
            valid_actions_list: [int],
            s_1: float,
            s_2: float,
            targeted: bool):

        if targeted:
            p_a_targeted = 1
            if h.trump == -1:  # trump phase
                if h.forehand == -1:
                    if m.forehand == 1:
                        a = (m.trump + TRUMP_FULL_OFFSET) if m.trump > -1 else -1
                    elif m.forehand == 0:
                        a = TRUMP_FULL_P
                    else:
                        a = -1
                else:
                    a = m.trump + TRUMP_FULL_OFFSET
            else:  # card phase
                a = m.tricks.reshape(-1)[h.nr_played_cards]

            if a == -1:  # action lies beyond current game history m
                a, p_a_targeted = self.sample_action(avg_strategy, valid_actions_list)

            p_a_untargeted = avg_strategy[a]
            return a, s_1 * p_a_targeted, s_2 * p_a_untargeted
        else:
            a, p_a_untargeted = self.sample_action(avg_strategy, valid_actions_list)

            if h.trump == -1:  # trump phase
                if h.forehand == -1:
                    if m.forehand == 1:
                        p_a_targeted = int(a == (m.trump + TRUMP_FULL_OFFSET))
                    elif m.forehand == 0:
                        p_a_targeted = int(a == TRUMP_FULL_P)
                    else:
                        p_a_targeted = 1
                else:
                    if m.forehand > -1:
                        p_a_targeted = int(a == (m.trump + TRUMP_FULL_OFFSET))
                    else:
                        p_a_targeted = 1
            else:  # card phase
                m_played_cards = m.tricks.reshape(-1)
                if m_played_cards[h.nr_played_cards] > -1:
                    p_a_targeted = int(a == m_played_cards[h.nr_played_cards])
                else:
                    p_a_targeted = 1

            return a, s_1 * p_a_targeted, s_2 * p_a_untargeted

    def sample_action(self, strategy, valid_actions_list):
        p = strategy[valid_actions_list]
        p /= p.sum()

        a = np.random.choice(valid_actions_list, p=p)
        p_a = strategy[a]
        return a, p_a

    def playout(self, h, a, l):
        while True:
            try:
                game = GameSimCpp()
                game.state = h
                game.perform_action_full(a)

                x = 1
                aa = []
                while not game.is_done():
                    valid_actions = np.flatnonzero(game.get_valid_actions())
                    assert len(valid_actions) > 0, f"invalid nr of valid actions {valid_actions}, {game.state}"
                    a = np.random.choice(valid_actions)
                    aa.append((a, valid_actions, game.state.player, game.state.hands.sum()))
                    game.perform_action_full(a)
                    x *= 1 / len(valid_actions)
            except AssertionError as e:
                logging.warning(f"caught: {e}, continuing anyways")
                continue

            break

        u = game.state.points
        return x, l * x, u

    def add_exploration(self, strategy, valid_actions_list):
        probs = strategy[valid_actions_list]
        probs = self.epsilon * 1 / len(valid_actions_list) + (1 - self.epsilon) * probs

        strategy[valid_actions_list] = probs
        strategy /= strategy.sum()

        assert np.isclose(strategy.sum(), 1)

        return strategy

    def regret_matching(self, regrets, valid_actions):
        positive_regrets = np.maximum(regrets, np.zeros_like(regrets))
        sum_pos_regret = positive_regrets.sum()
        if sum_pos_regret <= 0:
            strategy = np.zeros_like(regrets)
            strategy[valid_actions] = 1 / len(valid_actions)
        else:
            strategy = self.gamma / len(valid_actions) + (1 - self.gamma) * positive_regrets / sum_pos_regret
            strategy /= strategy.sum()
        return strategy

    @staticmethod
    def get_utility_for(points, i):
        return points[team[i]] - points[team[next_player[i]]]   # make utility zero sum


