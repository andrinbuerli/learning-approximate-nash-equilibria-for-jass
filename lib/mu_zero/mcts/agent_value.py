import jasscpp
import numpy as np
from jasscpp import GameObservationCpp

from lib.jass.agent.remembering_agent import RememberingAgent
from lib.jass.features.features_cpp_conv_cheating import FeaturesSetCppConvCheating
from lib.jass.features.features_set_cpp import FeaturesSetCpp
from lib.mu_zero.mcts.latent_node_selection_policy import LatentNodeSelectionPolicy
from lib.mu_zero.mcts.latent_value_calc_policy import LatentValueCalculationPolicy
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.tree_search import ALPV_MCTS
from lib.mu_zero.network.network_base import AbstractNetwork
from lib.mu_zero.network.support_conversion import support_to_scalar, support_to_scalar_per_player


class AgentValue(RememberingAgent):
    """
    Agent to play perfect imperfect information Jass with c++ ALPV-MCTS
    """

    def __init__(self,
                 network: AbstractNetwork,
                 feature_extractor: FeaturesSetCpp,
                 mdp_value:bool,
                 temperature=1.0):
        super().__init__(temperature=temperature)
        self.mdp_value = mdp_value
        self.network = network
        self.feature_extractor = feature_extractor
        self.cheating_mode = type(feature_extractor) == FeaturesSetCppConvCheating
        self.rule = jasscpp.RuleSchieberCpp()

    def get_play_action_probs_and_value(self, obs: GameObservationCpp, feature_format=None) -> np.array:
        features = self.feature_extractor.convert_to_features(obs, self.rule)

        valid_actions = self.rule.get_full_valid_actions_from_obs(obs)

        value, reward, policy, encoded_state = self.network.initial_inference(features[None])

        actions = np.flatnonzero(valid_actions).reshape(-1, 1)
        encoded_state = np.tile(encoded_state, (actions.shape[0], 1, 1, 1))
        value, reward, policy, encoded_state = self.network.recurrent_inference(encoded_state, actions)

        nr_players = value.shape[-2]
        value_support_size = value.shape[-1]
        value = support_to_scalar_per_player(value, min_value=-value_support_size//2, nr_players=nr_players)
        reward_support_size = reward.shape[-1]
        reward = support_to_scalar_per_player(reward, min_value=-reward_support_size//2, nr_players=nr_players)

        values = np.zeros(43)
        values[actions.reshape(-1)] = (value + reward)[:, obs.player].numpy().reshape(-1)

        values /= values.sum()

        return values, np.ones_like(values)
