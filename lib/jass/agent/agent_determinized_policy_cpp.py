# HSLU
#
# Created by Thomas Koller on 27.10.2020
#

import numpy as np
from jass.game.const import PUSH_ALT, PUSH
from jasscpp import RuleSchieberCpp, GameObservationCpp

from lib.jass.agent.agent import CppAgent
from lib.jass.mcts.random_hand_distribution_policy import RandomHandDistributionPolicy


class AgentDeterminizedPolicyCpp(CppAgent):
    """
    DNN to determine trump DMCTS for playing a card (with random determinisation and rollout)
    """

    def __init__(self,
                 model_path: str,
                 determinizations):
        from jassmlcpp.feature import FeaturesSetConvFromState2Cpp
        from jassmlcpp.model import LearnedModelCpp

        self.determinizations = determinizations
        self.learned_model = LearnedModelCpp()
        self.learned_model.load_model(directory=model_path,
                                      input_op_name='serving_default_input_1',
                                      output_op_name='StatefulPartitionedCall',
                                      tag_name='serve')
        self.features = FeaturesSetConvFromState2Cpp()

        self.rule = RuleSchieberCpp()

    def action_play_card(self, obs: GameObservationCpp) -> int:
        return self.get_action(obs)

    def action_trump(self, obs: GameObservationCpp) -> int:
        action = self.get_action(obs) - 36
        if action == PUSH_ALT:
            action = PUSH

        return action

    def get_action(self, obs):
        batch = []

        distribution_strategy = RandomHandDistributionPolicy()
        distribution_strategy.set_observation(obs)
        for _ in range(self.determinizations):
            state = distribution_strategy.get_state()
            features = self.features.convert_to_features(state)
            batch.append(features)

        policy = self.learned_model.execute_model_many(np.stack(batch))

        valid_actions = self.rule.get_valid_cards_from_obs(obs)

        return int(np.argmax(policy.mean(axis=0) * valid_actions))
