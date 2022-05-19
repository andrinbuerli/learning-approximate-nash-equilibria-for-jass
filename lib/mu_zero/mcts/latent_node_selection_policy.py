# HSLU
#
# Created by Thomas Koller on 02.10.18
#
from typing import Union

import jasscpp
import numpy as np
from jass.game.const import next_player, TRUMP_FULL_OFFSET, TRUMP_FULL_P, card_values
from jass.game.rule_schieber import RuleSchieber

from lib.jass.features.features_set_cpp import FeaturesSetCpp
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.node import Node
from lib.mu_zero.network.network_base import AbstractNetwork
from lib.mu_zero.network.support_conversion import support_to_scalar


class LatentNodeSelectionPolicy:

    def __init__(
            self,
            c_1: float,
            c_2: float,
            feature_extractor: FeaturesSetCpp,
            network: AbstractNetwork,
            discount: float,
            dirichlet_eps: float = 0.25,
            dirichlet_alpha: float = 0.3,
            mdp_value: bool = False,
            synchronized: bool = False,
            use_player_function: bool = False,
            use_terminal_function: bool = False,
            debug: bool = False):
        self.use_terminal_function = use_terminal_function
        self.use_player_function = use_player_function
        self.mdp_value = mdp_value
        self.synchronized = synchronized
        self.discount = discount
        self.c_2 = c_2
        self.c_1 = c_1
        self.network = network
        self.debug = debug
        self.dirichlet_alpha = dirichlet_alpha * np.ones(43)
        self.dirichlet_eps = dirichlet_eps
        self.feature_extractor = feature_extractor
        self.rule = RuleSchieber()

    def tree_policy(
            self,
            observation: jasscpp.GameObservationCpp,
            node: Node,
            stats: MinMaxStats,
            virtual_loss=0,
            observation_feature_format=None) -> Node:
        while True:
            with node.lock: # ensures that node and children not currently locked, i.e. being expanded
                node.visits += virtual_loss

            valid_actions = node.valid_actions

            assert valid_actions.sum() > 0, 'Error in valid actions'

            children = node.children_for_action(valid_actions)

            assert len(children) > 0, f'Error no children for valid actions {valid_actions}, {vars(node)}'

            #with node.lock: # ensures that node and children not currently locked, i.e. being expanded
            exploration = np.array([self._exploration_term(x) for x in children])
            exploitation = np.array([self._exploitation_term(x, stats) for x in children])

            # max_exploitation = 157 if not self.mdp_value else exploitation.max()
            # min_exploitation = 0 if not self.mdp_value else exploitation.min()

            #exploitation = (exploitation - min_exploitation) / np.max([(max_exploitation - min_exploitation), 1])
            #exploitation = stats.normalize(exploitation)

            puct = exploitation + exploration
            i_max = np.argmax(puct)
            child = children[i_max]

            for c in children:
                c.avail += 1

            # is_terminal_state = child.parent.is_post_terminal is not None and child.is_post_terminal > 0.5

            if self.use_player_function:
                is_terminal_state = child.is_post_terminal > 0.5 if self.use_terminal_function else False
            else:
                is_terminal_state = child.next_player == -1

            with node.lock:
                with child.lock:
                    not_expanded = child.prior is None
                    if not_expanded or is_terminal_state:
                        if not_expanded:
                            child.visits += virtual_loss
                            child.value, child.reward, child.prior, child.predicted_player, _, child.is_post_terminal, child.hidden_state = \
                                self.network.recurrent_inference(node.hidden_state, np.array([[child.action]]), all_preds=True)
                            self._expand_node(child, observation, observation_feature_format)
                        break

            node = child

        return child

    def init_node(self, node: Node, observation: Union[jasscpp.GameStateCpp, jasscpp.GameObservationCpp], observation_feature_format=None):
        if node.is_root():
            rule = jasscpp.RuleSchieberCpp()
            if observation_feature_format is not None:
                node.valid_actions = observation_feature_format.valid_actions
            elif type(observation) == jasscpp.GameStateCpp:
                node.valid_actions = rule.get_full_valid_actions_from_state(observation)
            else:
                node.valid_actions = rule.get_full_valid_actions_from_obs(observation)

            assert (node.valid_actions >= 0).all(), 'Error in valid actions'

            if observation_feature_format is not None:
                features = observation[None]
            else:
                features = self.feature_extractor.convert_to_features(observation, rule)[None]
            node.value, node.reward, node.prior, node.predicted_player, _, node.is_post_terminal, node.hidden_state =\
                self.network.initial_inference(features, all_preds=True)
            self._expand_node(node, root_obs=observation, observation_feature_format=observation_feature_format)

            valid_idxs = np.where(node.valid_actions)[0]
            eta = np.random.dirichlet(self.dirichlet_alpha[:len(valid_idxs)])
            node.prior[valid_idxs] = (1 - self.dirichlet_eps) * node.prior[valid_idxs] + self.dirichlet_eps * eta

    def _exploration_term(self, child: Node):
        P_s_a = child.parent.prior[child.action]
        prior_weight = (np.sqrt(child.avail) / (1 + child.visits)) * (
                    self.c_1 + np.log((child.avail + self.c_2 + 1) / self.c_2))
        exploration_term = P_s_a * prior_weight

        return exploration_term

    def _exploitation_term(self, child: Node, stats: MinMaxStats):
        if child.visits > 0:
            with child.lock:
                if self.use_player_function:
                    next_player = child.parent.predicted_player.argmax()
                else:
                    next_player = child.parent.next_player

                q = (child.value_sum[next_player] / child.visits)
                assert len(child.reward.shape) == 1, f'shape: {child.reward.shape}'
                q_value = (child.reward[next_player] + self.discount * q) \
                    if self.mdp_value else q

                q_value = stats.normalize(q_value)
            #logging.info(q_normed)
        else:
            q_value = 0

        return q_value

    def _expand_node(self, node: Node, root_obs: jasscpp.GameObservationCpp, observation_feature_format):
        node.value, node.reward, node.prior, node.predicted_player, node.is_post_terminal = \
            [x.numpy().squeeze() for x in [node.value, node.reward, node.prior, node.predicted_player, node.is_post_terminal]]

        #assert node.next_player == node.predicted_player.argmax()

        value_support_size = node.value.shape[-1]
        node.value = support_to_scalar(distribution=node.value, min_value=-value_support_size//2).numpy()
        reward_support_size = node.reward.shape[-1]
        node.reward = support_to_scalar(distribution=node.reward, min_value=-reward_support_size//2).numpy()

        #[node.stats.update(v) for v in node.value]
        #node.stats.update(node.value[node.parent.predicted_player.argmax()])

        # add edges for all children
        for action in node.missing_actions(node.valid_actions):
            if self.use_player_function:
                #add one child edge
                node.add_child(
                    action=action,
                    next_player=-1,
                    pushed=-1,
                    trump=-1,
                    mask_invalid=False)
            else:
                if action < TRUMP_FULL_OFFSET:
                    nr_played_cards = len(node.cards_played)
                    next_state_is_at_start_of_trick = (nr_played_cards + 1) % 4 == 0
                    if next_state_is_at_start_of_trick:
                        next_player_in_game = self._get_start_trick_next_player(action, node, root_obs, observation_feature_format)
                    else:
                        next_player_in_game = next_player[node.next_player]
                    node.add_child(
                        action=action,
                        next_player=next_player_in_game,
                        cards_played=list(node.cards_played + [action]))
                else:
                    trump = -1
                    pushed = None
                    if action == TRUMP_FULL_P:  # PUSH
                        next_player_in_game = (node.next_player + 2) % 4
                        pushed = True
                    else:  # TRUMP
                        if node.pushed:
                            next_player_in_game = (node.next_player + 2) % 4
                        else:
                            next_player_in_game = node.next_player
                        trump = action - 36
                    node.add_child(
                        action=action,
                        next_player=next_player_in_game,
                        pushed=pushed,
                        trump=trump) # mask push if played

    def _get_start_trick_next_player(self, action, node, root_obs, observation_feature_format):
        assert node.trump > -1

        prev_actions = [action]
        prev_values = [card_values[node.trump, action]]
        players = [node.next_player]
        parent = node

        if observation_feature_format is None:
            tricks = root_obs.tricks
            trick_first_player = root_obs.trick_first_player
        else:
            tricks = observation_feature_format.tricks
            trick_first_player = observation_feature_format.trick_first_player

        while parent.parent is not None and len(prev_actions) < 4:
            prev_actions.append(parent.action)
            prev_values.append(card_values[node.trump, parent.action])
            players.append(parent.player)
            parent = parent.parent

        num_cards = len(prev_actions)
        current_trick = len(node.cards_played) // 4
        for i in range(4 - num_cards):
            j = 4 - 1 - num_cards - i
            card = tricks[current_trick][j]
            prev_actions.append(card)
            prev_values.append(card_values[node.trump, card])
            players.append((trick_first_player[current_trick] - j) % 4)

        assert sum(players) == sum([0, 1, 2, 3]) and len(players) == 4, "invalid previous players"
        assert len(prev_actions) == 4 and len(prev_values) == 4, "invalid previous cards"

        next_player = self.rule.calc_winner(np.array(prev_actions[::-1]), players[-1], trump=node.trump)
        assert 0 <= next_player <= 3, "invalid next player"
        return next_player