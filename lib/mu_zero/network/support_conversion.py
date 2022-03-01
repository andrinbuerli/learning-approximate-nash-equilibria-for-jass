import tensorflow as tf


def support_to_scalar(distribution, min_value):
    """
    Transform a categorical representation to a scalar
    Calculate expectation over possible values
    """

    assert len(distribution.shape) == 2, "distribution must be batched"

    assert all(tf.abs(tf.reduce_sum(distribution, axis=1) - 1) < 1e-5), "probabilities do not sum up to 1"

    support_size = distribution.shape[1]
    indices = tf.range(start=min_value, limit=min_value + support_size, delta=1, dtype=tf.float32)

    expected_values = tf.reduce_sum(indices * distribution, axis=1)

    return expected_values


def scalar_to_support(scalar, support_size, min_value):
    """
    Transform a scalar to a categorical representation
    Only discrete values are assumed
    """

    assert len(scalar.shape) == 2, "scalar must be batched"

    assert (scalar % 1 == 0), "scalar must be integer"

    scalar = tf.clip_by_value(scalar, clip_value_min=min_value, clip_value_max=min_value + support_size)

    distribution = tf.one_hot(scalar, depth=support_size)

    return distribution