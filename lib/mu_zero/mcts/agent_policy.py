import jasscpp
import numpy as np
from jasscpp import GameObservationCpp

from lib.jass.agent.remembering_agent import RememberingAgent
from lib.jass.features.features_cpp_conv_cheating import FeaturesSetCppConvCheating
from lib.jass.features.features_set_cpp import FeaturesSetCpp
from lib.mu_zero.network.network_base import AbstractNetwork


class AgentPolicy(RememberingAgent):
    """
    Local policy agent to play the Schieber jass
    """

    def __init__(self,
                 network: AbstractNetwork,
                 feature_extractor: FeaturesSetCpp,
                 temperature=1.0):
        super().__init__(temperature=temperature)
        self.network = network
        self.feature_extractor = feature_extractor
        self.cheating_mode = type(feature_extractor) == FeaturesSetCppConvCheating
        self.rule = jasscpp.RuleSchieberCpp()


    def get_play_action_probs_and_values(self, obs: GameObservationCpp, feature_format=None) -> np.array:
        features = self.feature_extractor.convert_to_features(obs, self.rule)
        value, reward, policy, next_encoded_state = self.network.initial_inference(features[None])

        if type(obs) == GameObservationCpp:
            policy = policy.numpy().reshape(-1) * self.rule.get_full_valid_actions_from_obs(obs)
        else:
            policy = policy.numpy().reshape(-1) * self.rule.get_full_valid_actions_from_state(obs)

        values = np.ones_like(policy) # using only the local policy does not calculate a value
        return policy, values
