# HSLU
#
# Created by Thomas Koller on 02.10.18
#
import jasscpp
import numpy as np
from jass.game.const import next_player, TRUMP_FULL_OFFSET, TRUMP_FULL_P
from jasscpp import RuleSchieberCpp

from lib.jass.features.features_set_cpp import FeaturesSetCpp
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.node import Node
from lib.mu_zero.network.network_base import AbstractNetwork
from lib.mu_zero.network.support_conversion import support_to_scalar


class UCBLatentNodeSelectionPolicy:

    def __init__(
            self,
            c_1: float,
            c_2: float,
            feature_extractor: FeaturesSetCpp,
            network: AbstractNetwork,
            stats: MinMaxStats,
            discount: float,
            dirichlet_eps: float = 0.25,
            dirichlet_alpha: float = 0.3,
            synchronized: bool = False,
            debug: bool = False):
        self.synchronized = synchronized
        self.discount = discount
        self.stats = stats
        self.c_2 = c_2
        self.c_1 = c_1
        self.network = network
        self.debug = debug
        self.dirichlet_alpha = dirichlet_alpha * np.ones(43)
        self.dirichlet_eps = dirichlet_eps
        self.feature_extractor = feature_extractor
        self.rule = RuleSchieberCpp()

        self.nr_played_cards_in_selected_node = []

    def tree_policy(self, node: Node, virtual_loss=0) -> Node:
        while True:

            node.visits += virtual_loss

            valid_actions = node.valid_actions

            assert valid_actions.sum() > 0, 'Error in valid actions'

            # add edges for all children if any missing
            for action in node.missing_actions(valid_actions):
                # add one child edge
                if action < TRUMP_FULL_OFFSET:
                    next_player_in_game = next_player[node.player]
                elif action == TRUMP_FULL_P:  # PUSH
                    next_player_in_game = (node.player + 2) % 4
                else:  # TRUMP
                    next_player_in_game = node.player

                node.add_child(
                    action=action,
                    next_player=next_player_in_game)

            children = node.children_for_action(valid_actions)

            assert len(children) > 0, 'Error no children for valid actions {} '.format(valid_actions)

            child = max(children, key=self._puct(self.c_1, self.c_2))
            for c in children:
                c.avail += 1

            if self.synchronized:
                child.lock.acquire()
            not_expanded = child.prior is None
            if not_expanded:
                child.visits += virtual_loss
                child.value, child.reward, child.prior, child.hidden_state = \
                    self.network.recurrent_inference(node.hidden_state, np.array([[child.action]]))
                self._convert_node(child)
                if self.synchronized:
                    child.lock.release()
                break

            if self.synchronized:
                child.lock.release()
            node = child

        return child

    def init_node(self, node: Node, observation: jasscpp.GameObservationCpp):
        if node.is_root():
            rule = jasscpp.RuleSchieberCpp()
            node.valid_actions = rule.get_full_valid_actions_from_obs(observation)

            features = self.feature_extractor.convert_to_features(observation, rule)[None]
            node.value, node.reward, node.prior, node.hidden_state = self.network.initial_inference(features)
            self._convert_node(node)

            valid_idxs = np.where(node.valid_actions)[0]
            eta = np.random.dirichlet(self.dirichlet_alpha[:len(valid_idxs)])
            node.prior[valid_idxs] = (1 - self.dirichlet_eps) * node.prior[valid_idxs] + self.dirichlet_eps * eta


    def _puct(self, c_1: float, c_2: float):
        def f(child: Node) -> float:
            P_s_a = child.parent.prior[child.action]
            prior_weight = (np.sqrt(child.avail) / (1 + child.visits)) * (c_1 + np.log((child.avail + c_2 + 1) / c_2))
            exploration_term = P_s_a * prior_weight

            if child.visits > 0:
                exploitation_term = self.stats.normalize(
                    child.reward[child.player] + self.discount * child.exploitation_term)
            else:
                exploitation_term = 0

            return exploitation_term + exploration_term
        return f

    def _convert_node(self, node):
        node.value, node.reward, node.prior = \
            [x.numpy().squeeze() for x in [node.value, node.reward, node.prior]]

        node.value = support_to_scalar(distribution=node.value, min_value=0)
        node.reward = support_to_scalar(distribution=node.reward, min_value=-(node.reward.shape[1] // 2))