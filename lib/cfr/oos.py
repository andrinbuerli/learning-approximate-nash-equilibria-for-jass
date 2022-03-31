import logging

import numpy as np
from jass.game.const import next_player, TRUMP_FULL_OFFSET, TRUMP_FULL_P, team
from jasscpp import GameStateCpp, GameObservationCpp, RuleSchieberCpp, GameSimCpp

from lib.cfr.game_util import deal_random_hand


class OOS:

    def __init__(
            self,
            delta: float,
            epsilon: float,
            gamma: float,
            action_space: int,
            players: int,
            log:bool = False):
        self.log = log
        self.players = players
        self.gamma = gamma
        self.action_space = action_space
        self.epsilon = epsilon
        self.delta = delta
        self.infostates = {}  # infostate keys -> [cum. regrets, avg strategy, imm. regrets]
        self.rule = RuleSchieberCpp()

        self.immregrets = []

    def get_infostate_key(self, h: GameStateCpp):
        return f"{h.trump}-{h.forehand}-{h.hands[h.player]}-{h.tricks[h.current_trick]}"

    def get_infostate_key_from_obs(self, h: GameObservationCpp):
        return f"{h.trump}-{h.forehand}-{h.hand}-{h.tricks[h.current_trick]}"

    def run_iterations(self, m: GameObservationCpp, iterations: int, targeted_mode_init: bool = None):
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
                        if key in self.infostates:
                            p_a = self.infostates[key][1][a]
                        else:
                            p_a = 1 / game.get_valid_actions().sum()
                        game.perform_action_full(a)
                        p_bar_h *= p_a

                    a = m.trump + TRUMP_FULL_OFFSET
                    key = self.get_infostate_key(game.state)
                    if key in self.infostates:
                        p_a = self.infostates[key][1][a]
                    else:
                        p_a = 1 / game.get_valid_actions().sum()
                    game.perform_action_full(a)
                    p_bar_h *= p_a

                for c in m.tricks.reshape(-1):
                    if c == -1:
                        break

                    key = self.get_infostate_key(game.state)
                    if key in self.infostates:
                        p_a = self.infostates[key][1][c]
                    else:
                        p_a = 1 / game.get_valid_actions().sum()

                    game.perform_action_full(c)
                    p_bar_h *= p_a

                w_T = (1 - self.delta) + self.delta * p_bar_h
            else:
                w_T = 1

            for i in range(self.players):
                state = GameStateCpp()
                state.dealer = m.dealer
                state.player = next_player[m.dealer]
                self.iterate(m, state, 1, 1, w_T, w_T, i, targeted_mode)

            if self.log:
                #imregret = np.nanmean([r.max() for key, (_, _, r) in self.infostates.items()])
                key = self.get_infostate_key_from_obs(m)

                if key in self.infostates:
                    _, _, r = self.infostates[key]
                    imregret = r.max()
                else:
                    imregret = -1

                self.immregrets.append(imregret)
                logging.info(f"Touched infosets: {len(self.infostates)}, Average Regret: {imregret}")

    def iterate(
            self,
            m: GameObservationCpp,
            h: GameStateCpp,
            pi_i: float,
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
        is_terminal_history = h.nr_played_cards == 36
        is_chance_node = h.hands.min() == -1
        if is_terminal_history:
            return 1, self.delta * s_1 + (1 - self.delta) * s_2, h.points
        elif is_chance_node:
            known_hands = self.get_known_hands(m)
            hands_targeted, prob_targeted = deal_random_hand(known_hands=known_hands)
            known_hands = [hand if i == m.player else [] for i, hand in enumerate(known_hands)]
            hands_untargeted, prob_untargeted = deal_random_hand(known_hands=known_hands)

            if targeted_mode:
                h.hands = hands_targeted
            else:
                h.hands = hands_untargeted

            return self.iterate(
                m, h,
                pi_i, prob_untargeted * pi_o,
                prob_targeted * s_1, prob_untargeted * s_2,
                i, targeted_mode)

        infoset_key = self.get_infostate_key(h)
        a, s_1_prime, s_2_prime = self.sample(h, m, infoset_key, i, s_1, s_2, targeted_mode)

        valid_actions = self.rule.get_full_valid_actions_from_state(h)
        valid_actions_list = np.flatnonzero(valid_actions)
        if infoset_key not in self.infostates:
            current_strategy = valid_actions / valid_actions.sum()
            regret = np.zeros(self.action_space)
            avg_strategy = current_strategy
            imm_regrets = np.zeros_like(regret)
            self.infostates[infoset_key] = (
                regret, avg_strategy, imm_regrets
            )
            x, l, u = self.playout(h, a, (self.delta * s_1 + (1 - self.delta) * s_2) / valid_actions.sum())
        else:
            regret, avg_strategy, imm_regrets = self.infostates[infoset_key]
            current_strategy = self.regret_matching(regret, valid_actions_list)

            pi_i = current_strategy[a] * pi_i if h.player == i else pi_i
            pi_o = current_strategy[a] * pi_o if h.player != i else pi_o
            game.state = h
            game.perform_action_full(a)
            h_prime = game.state
            x, l, u = self.iterate(m, h_prime, pi_i, pi_o, s_1_prime, s_2_prime, i, targeted_mode)

        c = x
        x = x * current_strategy[a]

        for v_a in valid_actions_list:
            if h.player == i:
                W = u[team[i]] * pi_o / l
                if v_a == a:
                    imm_regrets[v_a] = (c - x) * W
                    regret[v_a] = regret[v_a] + imm_regrets[v_a]
                else:
                    imm_regrets[v_a] = - x * W
                    regret[v_a] = regret[v_a] + imm_regrets[v_a]
            else:
                avg_strategy[v_a] = avg_strategy[v_a] + 1 / (self.delta * s_1 + (1 - self.delta) * s_2) * pi_o * current_strategy[v_a]

        self.infostates[infoset_key] = (regret, avg_strategy, imm_regrets)

        return x, l, u

    def get_known_hands(self, m):
        known_hands = [[] for _ in range(4)]
        for j, trick in enumerate(m.tricks):
            card_player = m.trick_first_player[j]
            for card in trick:
                if card == -1:
                    break
                known_hands[card_player].append(card)
                card_player = next_player[card_player]
        known_hands[m.player] = list(set(known_hands[m.player] + np.flatnonzero(m.hand).tolist()))
        return known_hands

    def sample(
            self,
            h: GameStateCpp,
            m: GameObservationCpp,
            infoset_key: str,
            i: int,
            s_1: float,
            s_2: float,
            targeted: bool):

        if infoset_key in self.infostates:
            regret, avg_strategy, _ = self.infostates[infoset_key]
            avg_strategy /= avg_strategy.sum()
        else:
            valid_actions = self.rule.get_full_valid_actions_from_state(h)
            avg_strategy = valid_actions / valid_actions.sum()

        if targeted:
            p_a_targeted = 1
            if h.trump == -1: # trump phase
                if h.forehand == -1:
                    if m.forehand == 1:
                        a = (m.trump + TRUMP_FULL_OFFSET) if m.trump > -1 else -1
                    elif m.forehand == 0:
                        a = TRUMP_FULL_P
                    else:
                        a = -1
                else:
                    a = m.trump + TRUMP_FULL_OFFSET
            else: # card phase
                a = m.tricks.reshape(-1)[h.nr_played_cards]

            if a == -1: # action lies beyond current game history m
                a, p_a_targeted = self.sample_action_using_profile(avg_strategy, h, i)

            p_a_untargeted = avg_strategy[a]
            return a, s_1 * p_a_targeted, s_2 * p_a_untargeted
        else:
            a, p_a_untargeted = self.sample_action_using_profile(avg_strategy, h, i)

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

    def sample_action_using_profile(self, avg_strategy, h, i):
        valid_actions = np.flatnonzero(self.rule.get_full_valid_actions_from_state(h))
        probs = avg_strategy[valid_actions]
        probs = self.epsilon * 1 / len(valid_actions) + (1 - self.epsilon) * probs if h.player == i else probs
        assert np.isclose(probs.sum(), 1)
        a_index = np.random.choice(range(len(valid_actions)), p=probs)
        p_a = probs[a_index]
        a = valid_actions[a_index]
        return a, p_a

    def playout(self, h, a, l):
        game = GameSimCpp()
        game.state = h
        game.perform_action_full(a)

        x = 1
        while not game.is_done():
            valid_actions = np.flatnonzero(game.get_valid_actions())
            a = np.random.choice(valid_actions)
            game.perform_action_full(a)
            x *= 1 / len(valid_actions)

        u = game.state.points
        return x, l * x, u

    def regret_matching(self, regrets, valid_actions):
        positive_regrets = np.maximum(regrets, np.zeros_like(regrets))
        sum_pos_regret = positive_regrets.sum()
        if sum_pos_regret <= 0:
            strategy = np.zeros_like(regrets)
            strategy[valid_actions] = 1 / len(valid_actions)
            return strategy
        else:
            return self.gamma / len(valid_actions) + (1 - self.gamma) * positive_regrets / sum_pos_regret
