import shutil

import numpy as np

from lib.mu_zero.network.resnet import MuZeroResidualNetwork


def test_shapes():
    testee = MuZeroResidualNetwork(
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

    encoded_state = testee.representation(np.random.uniform(0, 1, (1, 4, 9, 45)))
    assert encoded_state.shape == (1, 4, 9, 256)

    encoded_next_state, reward = testee.dynamics(encoded_state, action=np.array([[1]]))
    assert encoded_next_state.shape == (1, 4, 9, 256)
    assert reward.shape == (1, 4, 201)

    policy, value = testee.prediction(encoded_next_state)
    assert policy.shape == (1, 43)
    assert value.shape == (1, 4, 101)


def test_summary():
    testee = MuZeroResidualNetwork(
        observation_shape=(4, 9, 43),
        action_space_size=42,
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

    testee.summary()

    # assert console output



def test_get_weights():
    testee = MuZeroResidualNetwork(
        observation_shape=(4, 9, 43),
        action_space_size=42,
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

    w = testee.get_weight_list()

    assert len(w) > 0


def test_set_weights():
    testee = MuZeroResidualNetwork(
        observation_shape=(4, 9, 43),
        action_space_size=42,
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

    w = testee.get_weight_list()

    w[0][0][0][0][0] = 1000

    testee.set_weights_from_list(w)

    assert testee.get_weight_list()[0][0][0][0][0] == 1000


def test_save_and_load():
    testee = MuZeroResidualNetwork(
        observation_shape=(4, 9, 43),
        action_space_size=42,
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

    path = f"test{id(testee)}.pd"
    testee.save(path)

    w = testee.get_weight_list()

    w[0][0][0][0][0] = 1000

    testee.set_weights_from_list(w)

    assert testee.get_weight_list()[0][0][0][0][0] == 1000

    testee.load(path)

    assert testee.get_weight_list()[0][0][0][0][0] != 1000

    shutil.rmtree(path)

