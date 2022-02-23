import jasscpp

from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.node import Node
from lib.mu_zero.mcts.ucb_latent_node_selection_policy import UCBLatentNodeSelectionPolicy
from lib.mu_zero.network.resnet import MuZeroResidualNetwork


def test_init():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 45),
        action_space_size=42,
        num_blocks=2,
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

    testee = UCBLatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            stats=MinMaxStats(),
            discount=1)

    node = Node(parent=None, action=None, player=None, next_player=1)
    testee.init_node(node, jasscpp.GameObservationCpp())

    assert node.prior is not None
    assert node.value is not None
    assert node.reward is not None
    assert node.hidden_state is not None
    assert node.valid_actions.sum() != node.valid_actions.shape[0]


def test_select():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 45),
        action_space_size=42,
        num_blocks=2,
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

    testee = UCBLatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            stats=MinMaxStats(),
            discount=1)

    node = Node(parent=None, action=None, player=1, next_player=1)
    testee.init_node(node, jasscpp.GameObservationCpp())

    child = testee.tree_policy(node)

    assert child.parent is node
    assert child.valid_actions.sum() == child.valid_actions.shape[0]

    assert node.prior is not None
    assert node.value is not None
    assert node.reward is not None
    assert node.hidden_state is not None
