
import numpy as np
from jasscpp import GameObservationCpp

from lib.jass.agent.remembering_agent import RememberingAgent
from lib.jass.features.features_set_cpp import FeaturesSetCpp
from lib.mu_zero.mcts.latent_node_selection_policy import LatentNodeSelectionPolicy
from lib.mu_zero.mcts.latent_value_calc_policy import LatentValueCalculationPolicy
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.tree_search import ALPV_MCTS
from lib.mu_zero.network.network_base import AbstractNetwork


class AgentMuZeroMCTS(RememberingAgent):
    """
    Agent to play perfect imperfect information Jass with c++ ALPV-MCTS
    """

    def __init__(self,
                 network: AbstractNetwork,
                 feature_extractor: FeaturesSetCpp,
                 iterations: int,
                 c_1=1,
                 c_2=19652,
                 dirichlet_eps=0.25,
                 dirichlet_alpha=0.3,
                 temperature=1.0,
                 discount=1,
                 mdp_value=False,
                 virtual_loss=10,
                 n_search_threads=4
                 ):
        super().__init__(temperature=temperature)

        self.iterations = iterations
        self.virtual_loss = virtual_loss
        self.mdp_value = mdp_value
        self.discount = discount
        self.n_search_threads = n_search_threads

        self.tree_policy = LatentNodeSelectionPolicy(
            c_1=c_1,
            c_2=c_2,
            feature_extractor=feature_extractor,
            network=network,
            synchronized=True,
            dirichlet_eps=dirichlet_eps,
            dirichlet_alpha=dirichlet_alpha,
            discount=discount)

        self.reward_calc_policy = LatentValueCalculationPolicy()


    def get_play_action_probs_and_value(self, obs: GameObservationCpp) -> np.array:
        stats = MinMaxStats()
        search = ALPV_MCTS(
            observation=obs,
            node_selection=self.tree_policy,
            reward_calc=self.reward_calc_policy,
            mdp_value=self.mdp_value,
            stats=stats,
            discount=self.discount,
            virtual_loss=self.virtual_loss,
            n_search_threads=self.n_search_threads
        )

        search.run_simulations_async(self.iterations)

        prob, q_value = search.get_result()

        return prob
