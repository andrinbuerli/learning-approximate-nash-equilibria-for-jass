
import numpy as np
from jasscpp import GameObservationCpp

from lib.cfr.oos import OOS
from lib.jass.agent.remembering_agent import RememberingAgent
from lib.jass.features.features_cpp_conv_cheating import FeaturesSetCppConvCheating
from lib.jass.features.features_set_cpp import FeaturesSetCpp
from lib.mu_zero.mcts.latent_node_selection_policy import LatentNodeSelectionPolicy
from lib.mu_zero.mcts.latent_value_calc_policy import LatentValueCalculationPolicy
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.tree_search import ALPV_MCTS
from lib.mu_zero.network.network_base import AbstractNetwork


class AgentOnlineOutcomeSampling(RememberingAgent):
    """
    Agent to play perfect imperfect information Jass 
    """

    def __init__(self,
                 iterations: int,
                 chance_samples: int,
                 delta=0.9,
                 epsilon=0.1,
                 gamma=0.01,
                 action_space=43,
                 players=4,
                 log=False,
                 chance_sampling=True,
                 temperature=1.0):
        super().__init__(temperature=temperature)

        self.cheating_mode = False

        if chance_sampling:
            self.iterations = iterations * chance_samples
        else:
            self.iterations = iterations

        self.search = OOS(
            delta=delta,
            epsilon=epsilon,
            gamma=gamma,
            action_space=action_space,
            players=players,
            chance_sampling=chance_sampling,
            iterations_per_chance_sample=iterations,
            log=log)

    def get_play_action_probs_and_value(self, obs: GameObservationCpp) -> np.array:

        self.search.run_iterations(obs, self.iterations)

        key = self.search.get_infostate_key_from_obs(obs)
        prob = self.search.get_average_stragety(key)

        # valid_actions = self.search.rule.get_full_valid_actions_from_obs(obs)
        # prob *= valid_actions
        # prob /= prob.sum()

        return prob

    def reset(self):
        self.search.reset()
