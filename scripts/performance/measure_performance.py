import time

import jasscpp
import numpy as np

from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.mu_zero.mcts.latent_value_calc_policy import LatentValueCalculationPolicy
from lib.mu_zero.mcts.min_max_stats import MinMaxStats
from lib.mu_zero.mcts.tree_search import ALPV_MCTS
from lib.mu_zero.mcts.latent_node_selection_policy import LatentNodeSelectionPolicy
from lib.mu_zero.network.buffering_network import BufferingNetwork
from lib.mu_zero.network.resnet import MuZeroResidualNetwork

if __name__ == "__main__":
    n = 5
    simulations = 100

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

    times = []
    for _ in range(n):
        stats = MinMaxStats()
        tree_policy = LatentNodeSelectionPolicy(
            c_1=1,
            c_2=19652,
            feature_extractor=FeaturesSetCppConv(),
            network=network,
            dirichlet_eps=0.25,
            dirichlet_alpha=0.3,
            discount=1)
        testee = ALPV_MCTS(
            observation=obs,
            node_selection=tree_policy,
            value_calc=LatentValueCalculationPolicy(),
            mdp_value=False,
            stats=stats,
            discount=1,
            virtual_loss=10
        )
        start = time.time()
        testee.run_simulations_sync(simulations)
        times.append(time.time() - start)
        print(f"\t\t {np.mean(times)}")
        del testee, tree_policy
    print(f"Using synch execution: {np.mean(times)}s")

    for n_search_threads in [1, 4, 8]:
        print("\n")
        times = []
        for _ in range(n):
            stats = MinMaxStats()
            tree_policy = LatentNodeSelectionPolicy(
                c_1=1,
                c_2=19652,
                feature_extractor=FeaturesSetCppConv(),
                network=network,
                dirichlet_eps=0.25,
                dirichlet_alpha=0.3,
                discount=1)

            testee = ALPV_MCTS(
                observation=obs,
                node_selection=tree_policy,
                value_calc=LatentValueCalculationPolicy(),
                mdp_value=False,
                stats=stats,
                discount=1,
                virtual_loss=10,
                n_search_threads=n_search_threads
            )

            start = time.time()
            testee.run_simulations_async(simulations)

            times.append(time.time() - start)
            print(f"\t\t {np.mean(times)}")
            del testee, tree_policy

        print(f"Using single execution, {n_search_threads} threads: {np.mean(times)}s")


    for n_search_threads in [1, 4, 8]:
        times = []
        for _ in range(n):
            stats = MinMaxStats()
            buffered_network = BufferingNetwork(network, buffer_size=n_search_threads, timeout=0.1)

            tree_policy = LatentNodeSelectionPolicy(
                c_1=1,
                c_2=100,
                feature_extractor=FeaturesSetCppConv(),
                network=buffered_network,
                dirichlet_eps=0.25,
                dirichlet_alpha=0.3,
                discount=1)

            testee = ALPV_MCTS(
                observation=obs,
                node_selection=tree_policy,
                value_calc=LatentValueCalculationPolicy(),
                mdp_value=False,
                stats=stats,
                discount=1,
                virtual_loss=10,
                n_search_threads=n_search_threads
            )

            start = time.time()
            testee.run_simulations_async(simulations)

            times.append(time.time() - start)
            print(f"\t\t {np.mean(times)}")
            del testee, tree_policy

        print(f"Using batched execution, {n_search_threads} threads: {np.mean(times)}s")

    del buffered_network