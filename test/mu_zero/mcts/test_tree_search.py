import jasscpp

from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.mu_zero.mcts.latent_value_calc_policy import LatentValueCalculationPolicy
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.node import Node
from lib.mu_zero.mcts.tree_search import ALPV_MCTS
from lib.mu_zero.mcts.ucb_latent_node_selection_policy import UCBLatentNodeSelectionPolicy
from lib.mu_zero.network.resnet import MuZeroResidualNetwork


def test_single_simulation():
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

    stats = MinMaxStats()

    tree_policy = UCBLatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            stats=stats,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        reward_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1
    )

    testee.run_simulation()

    assert testee.root.visits == 1


def test_multiple_simulations():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 45),
        action_space_size=43,
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

    stats = MinMaxStats()

    tree_policy = UCBLatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            stats=stats,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        reward_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1
    )

    testee.run_simulations(100)

    assert testee.root.visits == 100
