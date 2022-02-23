import numpy as np

import jasscpp

from lib.mu_zero.mcts.latent_value_calc_policy import LatentValueCalculationPolicy
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.node import Node
from lib.mu_zero.mcts.ucb_latent_node_selection_policy import UCBLatentNodeSelectionPolicy


class ALPV_MCTS:
    def __init__(self, observation: jasscpp.GameObservationCpp,
                 node_selection: UCBLatentNodeSelectionPolicy,
                 reward_calc: LatentValueCalculationPolicy,
                 stats: MinMaxStats,
                 mdp_value: bool = False,
                 discount: float = 0.9):
        """
        Initialize the search tree.
        Args:
            observation: the game observation at the start of the mcts search
            node_selection: the policy used for node selection and expansion
            reward_calc: the policy to calculate the reward from an expanded node
            mdp_value: True if the value calculation is n-step bootstrap, else value corresponds to game result estimation
            discount: Discount factor for n-step bootstrap value calculation
        """
        #
        self.stats = stats
        self.discount = discount
        self.mdp_value = mdp_value
        self.observation = observation

        # initialize root node
        # valid_actions = self.rule.get_valid_cards_from_state(self.state)

        # policies
        self.node_selection = node_selection
        self.reward_calc = reward_calc

        # initialize root node
        self.root = Node(parent=None, action=None, player=observation.player,
                         next_player=observation.player)

        self.node_selection.init_node(self.root, observation)

    def run_simulations(self, iterations: int) -> None:
        """
        Run a specified number of simulations.
        Args:
            iterations: the number of simulations to run.
        """

        # TODO implement multithreading
        for _ in range(iterations):
            self.run_simulation()

    def run_simulation(self) -> None:
        """
        Run one simulation from the root node
        """

        # select and possibly expand the tree using the tree policy
        node = self.node_selection.tree_policy(self.root)

        # evaluate the new node
        value = self.reward_calc.calculate_value(node)

        # back propagate the rewards from the last node
        while not node.is_root():
            node.propagate(value)

            if self.mdp_value:
                value = self.discount * value + node.reward
                self.stats.update(node.exploitation_term * self.discount + node.reward[node.player])
            else:
                self.stats.update(node.exploitation_term)

            node = node.parent

        node.propagate(value)

    def get_result(self) -> (np.ndarray, np.ndarray):
        """
        Get the (current) result of the simulations
        Returns:
            The probability of each action and the associated, estimated reward.
        """
        prob = np.zeros(43)
        reward = np.zeros(43)
        # there are different strategies here to select the best child
        # the current implementation is the most secure child :-)
        for action, node in self.root.children.items():
            prob[action] = node.visits
            reward[action] = node.rewards[node.player] / node.visits
        prob /= np.sum(prob)
        return prob, reward


