import numpy as np

from lib.mu_zero.network.buffering_network import BufferingNetwork
from lib.mu_zero.network.resnet import MuZeroResidualNetwork


def test_initial_inference():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 43),
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

    testee = BufferingNetwork(network, buffer_size=1)

    value, reward, policy, encoded_state = testee.initial_inference(np.random.uniform(0, 1, (1, 4, 9, 43)))
    assert encoded_state.shape == (1, 4, 9, 256)
    assert reward.shape == (1, 4, 201)
    assert policy.shape == (1, 42)
    assert value.shape == (1, 4, 101)

    del testee


def test_recurrent_inference():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 43),
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

    testee = BufferingNetwork(network, buffer_size=1)

    value, reward, policy, encoded_state = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]))
    assert encoded_state.shape == (1, 4, 9, 256)
    assert reward.shape == (1, 4, 201)
    assert policy.shape == (1, 42)
    assert value.shape == (1, 4, 101)

    del testee


def test_recurrent_inference_repeated_synch():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 43),
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

    testee = BufferingNetwork(network, buffer_size=1)

    value, reward, policy, encoded_state = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]))
    assert encoded_state.shape == (1, 4, 9, 256)
    assert reward.shape == (1, 4, 201)
    assert policy.shape == (1, 42)
    assert value.shape == (1, 4, 101)

    value, reward, policy, encoded_state = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]))
    assert encoded_state.shape == (1, 4, 9, 256)
    assert reward.shape == (1, 4, 201)
    assert policy.shape == (1, 42)
    assert value.shape == (1, 4, 101)

    del testee


def test_recurrent_inference_repeated_async():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 43),
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

    testee = BufferingNetwork(network, buffer_size=2)

    conn1 = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]), return_connection=True)
    conn2 = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]), return_connection=True)

    for conn in [conn1, conn2]:
        value, reward, policy, encoded_state = conn.recv()
        assert encoded_state.shape == (1, 4, 9, 256)
        assert reward.shape == (1, 4, 201)
        assert policy.shape == (1, 42)
        assert value.shape == (1, 4, 101)

    del testee


def test_recurrent_inference_too_small_buffer():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 43),
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

    testee = BufferingNetwork(network, buffer_size=2, timeout=10)

    conn = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]), return_connection=True)

    assert conn.poll(timeout=3.0) is False

    conn.close()

    del testee

def test_recurrent_inference_too_small_buffer_timeout():
    network = MuZeroResidualNetwork(
        observation_shape=(4, 9, 43),
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

    testee = BufferingNetwork(network, buffer_size=2, timeout=0.1)

    conn = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]), return_connection=True)

    assert conn.poll(timeout=3.0)

    conn.close()

    del testee