import shutil

import numpy as np

from lib.mu_zero.network.resnet import MuZeroResidualNetwork
from test.util import get_test_resnet


def test_shapes():
    testee = get_test_resnet()

    encoded_state = testee.representation(np.random.uniform(0, 1, (1, 4, 9, 45)))
    assert encoded_state.shape == (1, 4, 9, 256)

    encoded_next_state, reward = testee.dynamics(encoded_state, action=np.array([[1]]))
    assert encoded_next_state.shape == (1, 4, 9, 256)
    assert reward.shape == (1, 4, 101)

    policy, value = testee.prediction(encoded_next_state)
    assert policy.shape == (1, 43)
    assert value.shape == (1, 4, 101)


def test_summary():
    testee = get_test_resnet()

    testee.summary()

    # assert console output



def test_get_weights():
    testee = get_test_resnet()

    w = testee.get_weight_list()

    assert len(w) > 0


def test_set_weights():
    testee = get_test_resnet()

    w = testee.get_weight_list()

    w[0][0][0][0][0] = 1000

    testee.set_weights_from_list(w)

    assert testee.get_weight_list()[0][0][0][0][0] == 1000


def test_save_and_load():
    testee = get_test_resnet()

    path = f"test{id(testee)}.pd"
    testee.save(path)

    w = testee.get_weight_list()

    w[0][0][0][0][0] = 1000

    testee.set_weights_from_list(w)

    assert testee.get_weight_list()[0][0][0][0][0] == 1000

    testee.load(path)

    assert testee.get_weight_list()[0][0][0][0][0] != 1000

    shutil.rmtree(path)

