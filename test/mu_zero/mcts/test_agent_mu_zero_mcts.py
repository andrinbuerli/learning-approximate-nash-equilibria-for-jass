import jasscpp

from lib.factory import get_network
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.mu_zero.mcts.agent_mu_zero_mcts import AgentMuZeroMCTS
from lib.mu_zero.network.resnet import MuZeroResidualNetwork
from test.util import get_test_config


def test_prob_dist():
    config = get_test_config()
    network = get_network(config)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1

    testee = AgentMuZeroMCTS(
        network=network,
        feature_extractor=FeaturesSetCppConv(),
        iterations=100
    )

    dist, _ = testee.get_play_action_probs_and_values(obs)

    assert dist.shape[0] == 43


def test_play_card():
    config = get_test_config()
    network = get_network(config)

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
    config = get_test_config()
    network = get_network(config)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1

    testee = AgentMuZeroMCTS(
        network=network,
        feature_extractor=FeaturesSetCppConv(),
        iterations=20
    )

    for _ in range(5):
        action = testee.action_trump(obs)

    assert action < 7
