# HSLU
#
# Created by Thomas Koller on 02.10.18
#
from copy import deepcopy
from time import sleep

import jasscpp
import numpy as np
from jass.game.const import next_player, TRUMP_FULL_OFFSET, TRUMP_FULL_P
from jasscpp import RuleSchieberCpp

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
            debug: bool = False):
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
        self.rule = RuleSchieberCpp()

        self.nr_played_cards_in_selected_node = []

        self.stats = None

    def tree_policy(self, node: Node, stats: MinMaxStats, virtual_loss=0) -> Node:
        while True:
            with node.lock: # ensures that node and children not currently locked, i.e. being expanded
                node.visits += virtual_loss

            valid_actions = node.valid_actions

            assert valid_actions.sum() > 0, 'Error in valid actions'

            children = node.children_for_action(valid_actions)

            assert len(children) > 0, f'Error no children for valid actions {valid_actions}, {vars(node)}'

            with node.lock: # ensures that node and children not currently locked, i.e. being expanded
                child = max(children, key=lambda x: self._puct(x, stats))

            for c in children:
                c.avail += 1

            not_expanded = child.prior is None
            if not_expanded:
                with node.lock:
                    with child.lock:
                        child.visits += virtual_loss
                        child.value, child.reward, child.prior, child.hidden_state = \
                            self.network.recurrent_inference(node.hidden_state, np.array([[child.action]]))
                        self._expand_node(child)
                break

            node = child

        return child

    def init_node(self, node: Node, observation: jasscpp.GameObservationCpp):
        if node.is_root():
            rule = jasscpp.RuleSchieberCpp()
            node.valid_actions = rule.get_full_valid_actions_from_obs(observation)

            features = self.feature_extractor.convert_to_features(observation, rule)[None]
            node.value, node.reward, node.prior, node.hidden_state = self.network.initial_inference(features)
            self._expand_node(node)

            valid_idxs = np.where(node.valid_actions)[0]
            eta = np.random.dirichlet(self.dirichlet_alpha[:len(valid_idxs)])
            node.prior[valid_idxs] = (1 - self.dirichlet_eps) * node.prior[valid_idxs] + self.dirichlet_eps * eta


    def _puct(self, child: Node, stats: MinMaxStats):
        P_s_a = child.parent.prior[child.action]
        prior_weight = (np.sqrt(child.avail) / (1 + child.visits)) * (
                    self.c_1 + np.log((child.avail + self.c_2 + 1) / self.c_2))
        exploration_term = P_s_a * prior_weight

        if child.visits > 0:
            assert len(child.reward.shape) == 1, f'shape: {child.reward.shape}'
            exploitation_term = (child.reward[child.player] + self.discount * child.exploitation_term) \
                if self.mdp_value else child.exploitation_term
            exploitation_term_normed = stats.normalize(exploitation_term)
        else:
            exploitation_term_normed = 0

        return exploitation_term_normed + exploration_term

    def _expand_node(self, node):
        node.value, node.reward, node.prior = \
            [x.numpy().squeeze() for x in [node.value, node.reward, node.prior]]

        node.value = support_to_scalar(distribution=node.value, min_value=0).numpy()
        node.reward = support_to_scalar(distribution=node.reward, min_value=0).numpy()

        # add edges for all children
        for action in node.missing_actions(node.valid_actions):
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