import numpy as np


def support_to_scalar(distribution, min_value):
    """
    Transform a categorical representation to a scalar
    Calculate expectation over possible values
    """

    assert len(distribution.shape) == 2, "distribution must be batched"

    assert np.allclose(distribution.sum(axis=1), 1), "probabilities do not sum up to 1"

    support_size = distribution.shape[1]
    indices = np.arange(start=min_value, stop=min_value + support_size, step=1)

    expected_values = (indices * distribution).sum(axis=1)

    return expected_values


def scalar_to_support(scalar, support_size, min_value):
    """
    Transform a scalar to a categorical representation
    Only discrete values are assumed
    """

    assert len(scalar.shape) == 2, "scalar must be batched"

    assert (scalar % 1 == 0), "scalar must be integer"

    scalar = np.clip(scalar, a_min=min_value, a_max=min_value + support_size)

    distribution = np.zeros(support_size)
    distribution[scalar - min_value] = 1

    return distribution