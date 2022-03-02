import logging

import pytest

import numpy as np

from lib.mu_zero.replay_buffer.sum_tree import SumTree


def test_capacity():
    testee = SumTree(capacity=2)

    testee.add(p=.1, data=1)
    testee.add(p=.1, data=1)

    for _ in range(10):
        idx, p, data = testee.get(10)
        assert data == 1

    testee.add(p=1, data=2)
    testee.add(p=1, data=2)

    for _ in range(10):
        idx, p, data = testee.get(10)
        assert data == 2


def test_total():
    testee = SumTree(capacity=100)

    for i in range(100):
        testee.add(p=1, data=1)
        assert testee.total() == i + 1


def test_large_capacity():
    testee = SumTree(capacity=int(1e5))

    for i in range(int(1e4)):
        logging.info(i)
        value = np.random.uniform(0, 1, 1)
        testee.add(p=value, data=value)

    batches = []

    for _ in range(32):
        batch = []
        for __ in range(1024):
            s = np.random.uniform(0, testee.total())
            ___ = testee.get(s)
            batch.append(___)
        batches.append(batch)

    assert len(batches) == 32
    assert len(batches[0]) == 1024


@pytest.mark.skip(reason="Takes very long")
def test_large_capacity_with_update():
    testee = SumTree(capacity=int(1e5))

    for i in range(int(1.2e6)):
        logging.info(i)
        value = np.random.uniform(0, 1, 1)
        testee.add(p=value, data=value)

    sum_before = testee.total()

    for j in range(int(1e5)):
        total = testee.total()
        s = np.random.uniform(0, total)

        logging.info(f"{j}; total {total}; sample {s}")

        idx, priority, data = testee.get(s)
        testee.update(idx=idx, p=priority ** 2)

    sum_after = testee.total()

    assert sum_before > sum_after
