import time

import jasscpp

from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.mu_zero.mcts.latent_value_calc_policy import LatentValueCalculationPolicy
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.node import Node
from lib.mu_zero.mcts.tree_search import ALPV_MCTS
from lib.mu_zero.mcts.latent_node_selection_policy import LatentNodeSelectionPolicy
from lib.mu_zero.network.buffering_network import BufferingNetwork
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

    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
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

    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
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

    testee.run_simulations_sync(100)

    assert testee.root.visits == 100


def test_multiple_simulations_async_single_thread():
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

    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        reward_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1,
        virtual_loss=10,
        n_search_threads=1
    )

    testee.run_simulations_async(10)

    assert testee.root.visits == 10

def test_multiple_simulations_async_multi_thread():
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

    n_search_threads = 4
    buffered_network = BufferingNetwork(network, buffer_size=n_search_threads, timeout=0.1)

    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=buffered_network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        reward_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1,
        virtual_loss=10,
        n_search_threads=n_search_threads
    )

    testee.run_simulations_async(1000)

    assert testee.root.visits == 1000

    del buffered_network


def test_multiple_simulations_async_multi_thread_concurrency_check():
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

    n_search_threads = 4
    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        reward_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1,
        virtual_loss=0, # provoke concurrency issues
        n_search_threads=n_search_threads
    )

    testee.run_simulations_async(1000)

    assert testee.root.visits == 1000

    del testee


def test_get_rewards():
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

    n_search_threads = 4
    tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=100,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)

    obs = jasscpp.GameObservationCpp()
    obs.player = 1
    testee = ALPV_MCTS(
        observation=obs,
        node_selection=tree_policy,
        reward_calc=LatentValueCalculationPolicy(),
        mdp_value=False,
        stats=stats,
        discount=1,
        virtual_loss=0, # provoke concurrency issues
        n_search_threads=n_search_threads
    )

    testee.run_simulations_async(1000)

    prob, q_value = testee.get_result()

    assert prob.shape == (43,)
    assert q_value.shape == (43,)
