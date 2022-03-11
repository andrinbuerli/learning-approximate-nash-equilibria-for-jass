import jasscpp

from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.mu_zero.mcts.agent_mu_zero_mcts import AgentMuZeroMCTS
from lib.mu_zero.network.resnet import MuZeroResidualNetwork


def test_prob_dist():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 45),
        action_space_size=43,
                num_blocks_representation=2,
        num_blocks_dynamics=2,
        num_blocks_prediction=2,
        num_channels=256,
        reduced_channels_reward=128,
        reduced_channels_value=1,
        reduced_channels_policy=128,
        fc_reward_layers=[256],
        fc_value_layers=[256],
        fc_policy_layers=[256],
        support_size=100,
        players=4
    )

    obs = jasscpp.GameObservationCpp()
    obs.player = 1

    testee = AgentMuZeroMCTS(
        network=network,
        feature_extractor=FeaturesSetCppConv(),
        iterations=100
    )

    dist = testee.get_play_action_probs_and_value(obs)

    assert dist.shape[0] == 43


def test_play_card():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 45),
        action_space_size=43,
                num_blocks_representation=2,
        num_blocks_dynamics=2,
        num_blocks_prediction=2,
        num_channels=256,
        reduced_channels_reward=128,
        reduced_channels_value=1,
        reduced_channels_policy=128,
        fc_reward_layers=[256],
        fc_value_layers=[256],
        fc_policy_layers=[256],
        support_size=100,
        players=4
    )

    obs = jasscpp.GameObservationCpp()
    obs.player = 1

    testee = AgentMuZeroMCTS(
        network=network,
        feature_extractor=FeaturesSetCppConv(),
        iterations=100
    )

    action = testee.action_play_card(obs)

    assert action < 36


def test_play_trump():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 45),
        action_space_size=43,
                num_blocks_representation=2,
        num_blocks_dynamics=2,
        num_blocks_prediction=2,
        num_channels=256,
        reduced_channels_reward=128,
        reduced_channels_value=1,
        reduced_channels_policy=128,
        fc_reward_layers=[256],
        fc_value_layers=[256],
        fc_policy_layers=[256],
        support_size=100,
        players=4
    )

    obs = jasscpp.GameObservationCpp()
    obs.player = 1

    testee = AgentMuZeroMCTS(
        network=network,
        feature_extractor=FeaturesSetCppConv(),
        iterations=100
    )

    action = testee.action_trump(obs)

    assert action < 7
