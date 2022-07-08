from typing import Tuple

import numpy as np
from jass.game.const import next_player, TRUMP_FULL_OFFSET, TRUMP_FULL_P, team
from jasscpp import GameStateCpp, GameObservationCpp, RuleSchieberCpp, GameSimCpp

from lib.cfr.game_util import deal_random_hand

class OOS:
    """
    Online outcome sampling (OOS) implementation for the Schieber jass.
    Original paper: https://dl.acm.org/doi/abs/10.5555/2772879.2772887
    """

    def __init__(
            self,
            delta: float,
            epsilon: float,
            gamma: float,
            action_space: int,
            players: int,
            log: bool = False,
            asserts: bool = False):
        """
        Initialise the search tree
        :param delta: Parameter controlling targeting of current subgame
        :param epsilon: Parameter controlling amount of exploration
        :param gamma: Parameter to control regret matching
        :param action_space: Size of action space
        :param players: Number of players
        :param log: Controls weather oos details should be returned for logging
        :param asserts: Controls weather asserts should be executed
        """

        self.asserts = asserts
        self.log = log
        self.players = players
        self.teams = players // 2
        self.gamma = gamma
        self.action_space = action_space
        self.epsilon = epsilon
        self.delta = delta
        self.information_sets = {}  # infostate keys -> [cum. regrets, avg strategy, imm. regrets, valid_actions, (s_m, s_sum, num)]
        self.rule = RuleSchieberCpp()

    def reset(self):
        self.information_sets = {}

    def get_infostate_key(self, h: GameStateCpp):
        return str([h.trump, h.forehand, *np.flatnonzero(h.hands[h.player]), *h.tricks[h.current_trick]])

    def get_infostate_key_from_obs(self, m: Tuple[GameStateCpp, GameObservationCpp]):
        if isinstance(m, GameObservationCpp):
            return str([m.trump, m.forehand, *np.flatnonzero(m.hand), *m.tricks[m.current_trick]])
        else:
            return self.get_infostate_key(m)

    def get_average_strategy(self, informationset_key: str):
        _, avg_strategy, _, valid_actions, _ = self.information_sets[informationset_key]
        strategy_sum = avg_strategy.sum()

        if strategy_sum > 0:
            return avg_strategy / strategy_sum
        else:
            return valid_actions / valid_actions.sum()

    def run_iterations(self, m: Tuple[GameStateCpp, GameObservationCpp], iterations: int):
        """
        Run OOS iterations
        :param m: The current history
        :param iterations: number of iterations to run
        :return: default nothing is returned
                if logging is enabled the received immediate regrets, x, l, u and w_T values are returned
        """
        # only for logging purposes
        immediate_regrets, xs, ls, us, w_Ts = [], [], [], [], []

        w_T = self.calculate_weighting_factor(m)
        m_key = self.get_infostate_key_from_obs(m)

        # In order for I(m) to be present in information sets dict
        force_targeted_mode = m_key not in self.information_sets

        for _ in range(iterations):
            self.run_iteration(m, w_T, force_targeted_mode, immediate_regrets, ls, us, w_Ts, xs)
            if force_targeted_mode:
                force_targeted_mode = m_key not in self.information_sets

        if self.log:
            return immediate_regrets, xs, ls, us, w_Ts

    def run_iteration(self, m, w_T, force_targeted_mode, immediate_regrets, ls, us, w_Ts, xs):
        """
        Run a single OOS iteration
        :param m: the current history
        :param w_T: The weighting factor of the current history (see paper page 5)
        :param force_targeted_mode: boolean indicating if targeted mode should be forced
        :param immediate_regrets: list to append sampled immediate regrets
        :param ls: list to append sampled leaf node root reach probabilities
        :param us: list to append utilities of sampled leaf nodes
        :param w_Ts: list to append weighing factors of current history
        :param xs: list to append sampled leaf node reach probabilities from current history
        :return:
        """

        if force_targeted_mode is None:
            targeted_mode = bool(np.random.choice([1, 0], p=[self.delta, 1 - self.delta]))
        else:
            targeted_mode = force_targeted_mode
        hands, prob_targeted, prob_untargeted = self.sample_chance_outcome(m, targeted_mode)

        for i_team in range(self.teams):
            state = GameStateCpp()
            state.dealer = m.dealer
            state.player = next_player[m.dealer]
            state.hands = hands
            x, l, u = self.recurse(
                m=m,
                h=state,
                pi_i=1,
                pi_o=prob_untargeted * 1,
                s_1=prob_targeted * w_T,
                s_2=prob_untargeted * w_T,
                i_team=i_team,
                targeted_mode=targeted_mode)

            if self.log:
                xs.append(x), ls.append(l), us.append(u), w_Ts.append(w_T)
        if self.log:
            key = self.get_infostate_key_from_obs(m)
            if isinstance(m, GameObservationCpp):
                valid_actions = np.flatnonzero(self.rule.get_full_valid_actions_from_obs(m))
            else:
                valid_actions = np.flatnonzero(self.rule.get_full_valid_actions_from_state(m))

            if key in self.information_sets:
                _, _, r, _, _ = self.information_sets[key]
                imregret = r[valid_actions]
            else:
                imregret = [-1 for _ in valid_actions]

            immediate_regrets.append(imregret)

    def recurse(
            self,
            m: Tuple[GameStateCpp, GameObservationCpp],
            h: GameStateCpp,
            pi_i: [float],
            pi_o: [float],
            s_1: float,
            s_2: float,
            i_team: int,
            targeted_mode: bool):
        """
        Calculate OOS rollout recursively
        :param m: known history in ongoing game
        :param h: history in game sample
        :param pi_i: strategy reach probability for update team
        :param pi_o: strategy reach probability for opposing team
        :param s_1: overall probability that current sample is generated in targeted mode
        :param s_2: overall probability that current sample is generated in untargeted mode
        :param i_team: update team
        :param targeted_mode: update mode
        :return: sampled leaf node reach probability from current history (x),
                 sampled leaf node reach probability from root node (l),
                 utility of sampled leaf node
        """

        if self.asserts:
            assert s_1 <= 1, f"invalid value for s_1: {s_1}"
            assert s_2 <= 1, f"invalid value for s_2: {s_2}"

        is_terminal_history = h.nr_played_cards >= 36
        if is_terminal_history:
            return 1, self.delta * s_1 + (1 - self.delta) * s_2, h.points

        infoset_key = self.get_infostate_key(h)

        if infoset_key in self.information_sets:
            regret, avg_strategy, imm_regrets, valid_actions, _ = self.information_sets[infoset_key]
            valid_actions_list = np.flatnonzero(valid_actions)

            if self.asserts:
                assert (valid_actions == self.rule.get_full_valid_actions_from_state(h)).all()

            current_strategy = self.regret_matching(regret, valid_actions)
            if team[h.player] == i_team:
                current_strategy = self.add_exploration(current_strategy, valid_actions_list)
        else:
            valid_actions = self.rule.get_full_valid_actions_from_state(h)
            valid_actions_list = np.flatnonzero(valid_actions)
            current_strategy = valid_actions / valid_actions.sum()

        a, s_1_prime, s_2_prime = self.sample(h, m, current_strategy, valid_actions_list, s_1, s_2, targeted_mode)

        if infoset_key not in self.information_sets:
            self.add_information_set(infoset_key, valid_actions)
            x, l, u = self.random_playout(h, a, current_strategy[a], (self.delta * s_1 + (1 - self.delta) * s_2) / valid_actions.sum())
        else:
            pi_i_prime =  current_strategy[a] * pi_i if team[h.player] == i_team else pi_i
            pi_o_prime = current_strategy[a] * pi_o if team[h.player] != i_team else pi_o

            h_prime = self.apply_action(a, h)
            x, l, u = self.recurse(m, h_prime, pi_i_prime, pi_o_prime, s_1_prime, s_2_prime, i_team, targeted_mode)

        c = x
        x = x * current_strategy[a]

        regret, avg_strategy, imm_regrets, valid_actions, (s_m, s_sum, num) = self.information_sets[infoset_key]
        for v_a in valid_actions_list:
            if team[h.player] == i_team:
                u_i = self.get_utility_for(u, i_team)
                W = u_i * pi_o / l
                if v_a == a:
                    imm_regrets[v_a] = (c - x) * W
                else:
                    imm_regrets[v_a] = - x * W

                regret[v_a] = regret[v_a] + imm_regrets[v_a]
            else:
                pi_i_I = (self.delta * s_1 + (1 - self.delta) * s_2)
                avg_strategy[v_a] = avg_strategy[v_a] + (pi_o / pi_i_I) * current_strategy[v_a]

        # only update s_sum if h is in current subgame
        if (m.trump == -1 and h.trump > -1) or (m.forehand == -1 and h.forehand > -1) or m.nr_played_cards < h.nr_played_cards:
            s_sum = s_sum + (self.delta * s_1 + (1 - self.delta) * s_2)
            num = num + 1

        self.information_sets[infoset_key] = (regret, avg_strategy, imm_regrets, valid_actions, (s_m, s_sum, num))

        return x, l, u

    def apply_action(self, a, h):
        game = GameSimCpp()
        game.state = h
        game.perform_action_full(a)
        h_prime = game.state
        return h_prime

    def calculate_weighting_factor(self, m):
        """
        Calculate the weighing factor of history based on it's in memory samples
        :param m: current history
        :return: the weighing factor w_T, inverted when compared to factor in paper on page 5
        """

        m_key = self.get_infostate_key_from_obs(m)
        if m_key not in self.information_sets:
            s_m = 1
            s_0 = s_m
        else:
            regret, avg_strategy, imm_regrets, valid_actions, (s_m, s_sum, num) = self.information_sets[m_key]
            s_0 = s_sum / max(num, 1)
            if s_m == 0:
                # The first time weighting factor is calculated
                # s_m is calculated as the probability of m being sampled under the current average strategy

                s_untargeted = 1
                known_hands = self.get_known_hands(m)

                _, prob_chance_targeted = deal_random_hand(known_hands=known_hands)
                if isinstance(m, GameObservationCpp):
                    known_hands = [hand if i == m.player else [] for i, hand in enumerate(known_hands)]
                hands_untargeted, prob_untargeted = deal_random_hand(known_hands=known_hands)

                s_untargeted *= prob_untargeted # can be factorized?

                game = GameSimCpp()
                game.state.hands = hands_untargeted
                game.state.dealer = m.dealer
                game.state.player = next_player[m.dealer]

                if m.forehand > -1:
                    if m.forehand == 0:
                        a = TRUMP_FULL_P
                        key = self.get_infostate_key(game.state)
                        if key in self.information_sets:
                            p_a = self.get_average_strategy(key)[a]
                        else:
                            p_a = 1 / game.get_valid_actions().sum()
                        game.perform_action_full(a)
                        s_untargeted *= p_a

                    if m.trump > -1:
                        a = m.trump + TRUMP_FULL_OFFSET
                        key = self.get_infostate_key(game.state)
                        if key in self.information_sets:
                            p_a = self.get_average_strategy(key)[a]
                        else:
                            p_a = 1 / game.get_valid_actions().sum()
                        game.perform_action_full(a)
                        s_untargeted *= p_a

                for c in m.tricks.reshape(-1):
                    if c == -1:
                        break

                    key = self.get_infostate_key(game.state)
                    if key in self.information_sets:
                        p_a = self.get_average_strategy(key)[c]
                    else:
                        p_a = 1 / game.get_valid_actions().sum()

                    game.perform_action_full(c)
                    s_untargeted *= p_a

                s_targeted = prob_chance_targeted
                s_m = self.delta * s_targeted + (1 - self.delta) * s_untargeted
                self.information_sets[m_key] = (regret, avg_strategy, imm_regrets, valid_actions, (s_m, s_sum, num))

            if s_0 == 0:  # information set has never been sampled before
                s_0 = s_m

        w_T_inv = s_0 / s_m

        return w_T_inv

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

        # add potentially missing information sets
        self.add_missing_information_sets(hands_targeted, m)

        if self.asserts:
            assert hands.sum() == 36

        return hands, prob_targeted, prob_untargeted

    def add_missing_information_sets(self, hands_targeted, m):
        """
        Given a sampled hand distribution and a game history, add all compatible information sets to memory
        :param hands_targeted: sampled hand distribution as a numpy array with shape (4, 9)
        :param m: the current history
        :return:
        """
        game = GameSimCpp()
        game.state.hands = hands_targeted
        game.state.dealer = m.dealer
        game.state.player = next_player[m.dealer]
        p = 1
        if m.forehand > -1:
            if m.forehand == 0:
                a = TRUMP_FULL_P
                key = self.get_infostate_key(game.state)
                if key not in self.information_sets:
                    self.add_information_set(key, game.get_valid_actions())
                p_a = self.get_average_strategy(key)[a]
                p *= p_a
                game.perform_action_full(a)

            a = m.trump + TRUMP_FULL_OFFSET
            key = self.get_infostate_key(game.state)
            if key not in self.information_sets:
                self.add_information_set(key, game.get_valid_actions())
            game.perform_action_full(a)
            p_a = self.get_average_strategy(key)[a]
            p *= p_a
        for c in m.tricks.reshape(-1):
            if c == -1:
                break

            key = self.get_infostate_key(game.state)
            if key not in self.information_sets:
                self.add_information_set(key, game.get_valid_actions())

            game.perform_action_full(c)
            p_a = self.get_average_strategy(key)[c]
            p *= p_a

    def add_information_set(self, informationset_key, valid_actions):
        regret = np.zeros(self.action_space)
        avg_strategy = np.zeros(self.action_space)
        imm_regrets = np.zeros(self.action_space)
        self.information_sets[informationset_key] = (
            regret, avg_strategy, imm_regrets, valid_actions, (0, 0, 0)
        )
        return avg_strategy, imm_regrets, regret

    def get_known_hands(self, m):
        """
        Get the known hand distribution from a history as an array
        :param m: the current history
        :return: known hand distribution as an array with shape (4, 9)
        """

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
            current_strategy: np.array,
            valid_actions_list: [int],
            s_1: float,
            s_2: float,
            targeted: bool):
        """
        Sample next action following the given strategy and targeting mode
        :param h: the current history in the oos iteration
        :param m: the current history in the actual game
        :param current_strategy: The strategy to use to select the next action
        :param valid_actions_list: list of valid actions
        :param s_1: probability that current sample is generated in targeted mode
        :param s_2: probability that current sample is generated in untargeted mode
        :param targeted: boolean indicating weather the oos iteration is in targeted or untargeted mode
        :return:
        """

        if targeted:
            p_a_targeted = 1
            if h.trump == -1 and m.trump > -1:  # trump phase
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
                a, p_a_targeted = self.sample_valid_action(current_strategy, valid_actions_list)

            p_a_untargeted = current_strategy[a]
            return a, s_1 * p_a_targeted, s_2 * p_a_untargeted
        else:
            a, p_a_untargeted = self.sample_valid_action(current_strategy, valid_actions_list)

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

    @staticmethod
    def sample_valid_action(strategy, valid_actions_list):
        p = strategy[valid_actions_list]
        p /= p.sum()

        a = np.random.choice(valid_actions_list, p=p)
        p_a = strategy[a]
        return a, p_a

    @staticmethod
    def random_playout(h, selected_action, p_a, l):
        """
        Run random playout of subgame rooted at history after selected action was performed
        :param h: the current history
        :param selected_action: the selected action to perform
        :param p_a: probability of sampling the selected action
        :param l: reach probability of current history from the root
        :return: reach probability of sampled leaf node from current history (x),
                 reach probability of sampled leaf node from root (l),
                 utility of sampled leaf node
        """

        game = GameSimCpp()
        game.state = h
        game.perform_action_full(selected_action)

        x = 1
        while not game.is_done():
            valid_actions = np.flatnonzero(game.get_valid_actions())
            assert len(valid_actions) > 0, f"invalid nr of valid actions {valid_actions}, {game.state}"
            a = np.random.choice(valid_actions)
            game.perform_action_full(a)
            x *= 1 / len(valid_actions)

        u = game.state.points
        return x, l * p_a * x, u

    def add_exploration(self, strategy, valid_actions_list):
        probs = strategy[valid_actions_list]
        probs = self.epsilon * 1 / len(valid_actions_list) + (1 - self.epsilon) * probs

        strategy[valid_actions_list] = probs
        strategy /= strategy.sum()

        if self.asserts:
            assert np.isclose(strategy.sum(), 1)

        return strategy

    def regret_matching(self, regrets, valid_actions):
        positive_regrets = np.maximum(regrets, np.zeros_like(regrets))
        sum_pos_regret = positive_regrets.sum()
        if sum_pos_regret <= 0:
            strategy = valid_actions
        else:
            strategy = valid_actions * self.gamma / valid_actions.sum() + (1 - self.gamma) * positive_regrets / sum_pos_regret
        strategy = strategy / strategy.sum()
        return strategy

    @staticmethod
    def get_utility_for(points, i_team):
        opposing_team = 1 - i_team
        return points[i_team] - points[opposing_team]   # make utility zero sum


