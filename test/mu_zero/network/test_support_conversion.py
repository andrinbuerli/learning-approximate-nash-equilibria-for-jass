import numpy as np
import pytest

from lib.mu_zero.network.support_conversion import support_to_scalar, scalar_to_support


def test_support_to_scalar():
    distribution = np.array([[0.1, 0.9]])

    scalar = support_to_scalar(distribution, min_value=0)

    assert scalar == 0.9

def test_support_to_scalar_min_value():
    distribution = np.array([[0.1, 0.9]])

    scalar = support_to_scalar(distribution, min_value=-1)

    assert scalar == -.1

def test_support_to_scalar_distribution_non_1_error():
    distribution = np.array([[0.0, 0.9]])

    with pytest.raises(AssertionError):
        support_to_scalar(distribution, min_value=-1)


def test_support_to_scalar_batched():
    distribution = np.array([
        [0.1, 0.9],
        [0.9, 0.1]
    ])

    scalar = support_to_scalar(distribution, min_value=-1)

    assert all(scalar == [-.1, -.9])


def test_scalar_to_support():
    scalar = np.array([[1]])

    distribution = scalar_to_support(scalar, min_value=0, support_size=2)

    assert all(distribution == [0, 1])


def test_scalar_to_support_non_integer_error():
    scalar = np.array([[1.1]])

    with pytest.raises(AssertionError):
        scalar_to_support(scalar, min_value=0, support_size=2)
