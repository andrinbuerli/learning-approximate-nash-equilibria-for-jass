import numpy as np
from multiprocessing.pool import ThreadPool

import jasscpp

from lib.mu_zero.mcts.latent_value_calc_policy import LatentValueCalculationPolicy
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.node import Node
from lib.mu_zero.mcts.latent_node_selection_policy import LatentNodeSelectionPolicy


class ALPV_MCTS:
    def __init__(self, observation: jasscpp.GameObservationCpp,
                 node_selection: LatentNodeSelectionPolicy,
                 reward_calc: LatentValueCalculationPolicy,
                 stats: MinMaxStats,
                 mdp_value: bool = False,
                 discount: float = 0.9,
                 n_search_threads: int = 4,
                 virtual_loss: int = 10,
                 store_trajectory_actions: bool = False):
        """
        Initialize the search tree.
        Args:
            observation: the game observation at the start of the mcts search
            node_selection: the policy used for node selection and expansion
            reward_calc: the policy to calculate the reward from an expanded node
            mdp_value: True if the value calculation is n-step bootstrap, else value corresponds to game result estimation
            discount: Discount factor for n-step bootstrap value calculation
            n_search_threads: Number of search threads for async simulations
            virtual_loss: Virtual loss temporary added to nodes in search thread
        """
        #
        self.store_trajectory_actions = store_trajectory_actions
        self.trajectory_actions = []
        self.n_search_threads = n_search_threads
        self.virtual_loss = virtual_loss
        self.stats = stats
        self.discount = discount
        self.mdp_value = mdp_value
        self.observation = observation

        # policies
        self.node_selection = node_selection
        self.reward_calc = reward_calc

        # initialize root node
        cards_played = [x for x in observation.tricks.reshape(-1).tolist() if x >= 0]
        self.root = Node(parent=None, action=None, player=None, trump=observation.trump,
                         next_player=observation.player, cards_played=cards_played)

        self.node_selection.init_node(self.root, observation)

        self.pool = ThreadPool(processes=self.n_search_threads)

    def run_simulations_sync(self, iterations: int) -> None:
        """
        Run a specified number of simulations.
        Args:
            iterations: the number of simulations to run.
        """

        for _ in range(iterations):
            self.run_simulation()

    def run_simulations_async(self, iterations: int) -> None:
        """
        Run a specified number of parallelized simulations.
        Args:
            iterations: the number of simulations to run.
        """

        self.pool.map(lambda _: self.run_simulation(), range(iterations))

    def run_simulation(self) -> None:
        """
        Run one simulation from the root node
        """

        # select and possibly expand the tree using the tree policy
        node = self.node_selection.tree_policy(node=self.root, observation=self.observation,
                                               virtual_loss=self.virtual_loss,
                                               stats=self.stats)

        # evaluate the new node
        value = self.reward_calc.calculate_value(node)

        actions = []

        # back propagate the rewards from the last node
        while True:
            actions.append(node.action)
            with node.lock:
                node.propagate(value, self.virtual_loss)

            if node.is_root():
                break

            q = (node.value_sum[node.player] / node.visits)
            if self.mdp_value:
                value = self.discount * value + node.reward
                self.stats.update(q * self.discount + node.reward[node.player])
            else:
                self.stats.update(q)


            node = node.parent

        if self.store_trajectory_actions:
            self.trajectory_actions.append(list(reversed(actions)))

    def get_result(self) -> (np.ndarray, np.ndarray):
        """
        Get the (current) result of the simulations
        Returns:
            The probability of each action and the associated, estimated reward.
        """
        prob = np.zeros(43)
        q_value = np.zeros(43)

        for action, node in self.root.children.items():
            prob[action] = node.visits
            q = (node.value_sum[node.player] / node.visits)
            if self.mdp_value:
                q_value[action] = q * self.discount + node.reward[node.player]
            else:
                q_value[action] = q
        prob /= np.sum(prob)
        return prob, q_value


    def __del__(self):
        if hasattr(self, "pool"):
            self.pool.terminate()


