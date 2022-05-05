import numpy as np

from lib.mu_zero.network.buffering_network import BufferingNetwork
from test.util import get_test_resnet


def test_initial_inference():
    network = get_test_resnet()

    testee = BufferingNetwork(network, buffer_size=1)

    value, reward, policy, encoded_state = testee.initial_inference(np.random.uniform(0, 1, (1, 4, 9, 43)))
    assert encoded_state.shape == (1, 4, 9, 256)
    assert reward.shape == (1, 4, 101)
    assert policy.shape == (1, 42)
    assert value.shape == (1, 4, 101)

    del testee


def test_recurrent_inference():
    network = get_test_resnet()

    testee = BufferingNetwork(network, buffer_size=1)

    value, reward, policy, encoded_state = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]))
    assert encoded_state.shape == (1, 4, 9, 256)
    assert reward.shape == (1, 4, 101)
    assert policy.shape == (1, 42)
    assert value.shape == (1, 4, 101)

    del testee


def test_recurrent_inference_repeated_synch():
    network = get_test_resnet()

    testee = BufferingNetwork(network, buffer_size=1)

    value, reward, policy, encoded_state = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]))
    assert encoded_state.shape == (1, 4, 9, 256)
    assert reward.shape == (1, 4, 101)
    assert policy.shape == (1, 42)
    assert value.shape == (1, 4, 101)

    value, reward, policy, encoded_state = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]))
    assert encoded_state.shape == (1, 4, 9, 256)
    assert reward.shape == (1, 4, 101)
    assert policy.shape == (1, 42)
    assert value.shape == (1, 4, 101)

    del testee


def test_recurrent_inference_repeated_async():
    network = get_test_resnet()

    testee = BufferingNetwork(network, buffer_size=2)

    conn1 = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]), return_connection=True)
    conn2 = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]), return_connection=True)

    for conn in [conn1, conn2]:
        value, reward, policy, encoded_state = conn.recv()
        assert encoded_state.shape == (1, 4, 9, 256)
        assert reward.shape == (1, 4, 101)
        assert policy.shape == (1, 42)
        assert value.shape == (1, 4, 101)

    del testee


def test_recurrent_inference_too_small_buffer():
    network = get_test_resnet()

    testee = BufferingNetwork(network, buffer_size=2, timeout=10)

    conn = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]), return_connection=True)

    assert conn.poll(timeout=3.0) is False

    conn.close()

    del testee

def test_recurrent_inference_too_small_buffer_timeout():
    network = get_test_resnet()

    testee = BufferingNetwork(network, buffer_size=2, timeout=0.1)

    conn = testee.recurrent_inference(np.random.uniform(0, 1, (1, 4, 9, 256)), np.array([[1]]), return_connection=True)

    assert conn.poll(timeout=5.0)

    conn.close()

    del testee