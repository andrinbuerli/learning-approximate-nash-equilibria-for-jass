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
                 store_trajectory_actions: bool = False,
                 observation_feature_format=None):
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

        self.observation_feature_format = observation_feature_format

        if observation_feature_format is None:
            cards_played = [x for x in observation.tricks.reshape(-1).tolist() if x >= 0]
            root_player = observation.player
            trump = observation.trump
        else:
            reshaped = observation.reshape(observation_feature_format.FEATURE_SHAPE)
            root_player = reshaped[0, 0, observation_feature_format.CH_PLAYER:observation_feature_format.CH_PLAYER + 4].argmax()
            trump_one_hot = reshaped[0, 0, observation_feature_format.CH_TRUMP:observation_feature_format.CH_TRUMP + 6]
            if trump_one_hot.max() > 0:
                trump = trump_one_hot.argmax()
            else:
                trump = -1

            valid_cards = reshaped[:, :, observation_feature_format.CH_CARDS_VALID].reshape(-1)
            trump_valid = np.repeat(reshaped[0, 0, observation_feature_format.CH_TRUMP_VALID].reshape(-1), 6)
            push_valid = reshaped[0, 0, observation_feature_format.CH_PUSH_VALID].reshape(-1)

            observation_feature_format.valid_actions = np.concatenate((valid_cards, trump_valid, push_valid))

            observation_feature_format.tricks = -np.ones((9, 4), dtype=np.int32)
            observation_feature_format.trick_first_player = -np.ones(9, dtype=np.int32)
            trick_cards_position = reshaped[:, :,
                                   observation_feature_format.CH_CARDS_IN_POSITION:observation_feature_format.CH_CARDS_IN_POSITION + 4]\
                .argmax(axis=-1).reshape(-1)
            trick_cards_player = reshaped[:, :,
                                 observation_feature_format.CH_CARDS_PLAYER_0:observation_feature_format.CH_CARDS_PLAYER_0 + 4]\
                .argmax(axis=-1).reshape(-1)
            for i in range(9):
                trick_cards_one_hot = reshaped[:, :, observation_feature_format.CH_CARDS_IN_TRICK + i].reshape(-1)
                trick_cards = np.flatnonzero(trick_cards_one_hot)

                if len(trick_cards) == 0:
                    break

                for c in trick_cards:
                    trick_card_position = trick_cards_position[c]
                    observation_feature_format.tricks[i, trick_card_position] = int(c)
                    if trick_card_position == 0:
                        observation_feature_format.trick_first_player[i] = int(trick_cards_player[c])

            cards_played = [int(x) for x in observation_feature_format.tricks.reshape(-1).tolist() if x > -1]

        self.root = Node(parent=None, action=None, player=None, trump=trump,
                         next_player=root_player, cards_played=cards_played)

        self.node_selection.init_node(self.root, observation, observation_feature_format)

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
                                               observation_feature_format=self.observation_feature_format,
                                               stats=self.stats)

        # evaluate the new node
        value = self.reward_calc.calculate_value(node)

        actions = []

        # back propagate the rewards from the last node
        while True:
            actions.append(node.action)
            with node.lock:
                node.propagate(value, self.virtual_loss)

            [self.stats.update(v) for v in value]

            if node.is_root():
                break

            if self.mdp_value:
                value = self.discount * value + node.reward

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
        q_value = np.zeros((43, 2))

        for action, child in self.root.children.items():
            prob[action] = child.visits
            team = child.player % 2
            q = (child.value_sum[team] / max(child.visits, 1))
            if self.mdp_value:
                q_value[action, team] = q * self.discount + child.reward[team]
            else:
                q_value[action, team] = q

            other_team = (team + 1) % 2
            q = (child.value_sum[other_team] / max(child.visits, 1))
            if self.mdp_value:
                q_value[action, other_team] = q * self.discount + child.reward[other_team]
            else:
                q_value[action, other_team] = q

        prob /= np.sum(prob)
        return prob, q_value


    def __del__(self):
        if hasattr(self, "pool"):
            self.pool.terminate()


