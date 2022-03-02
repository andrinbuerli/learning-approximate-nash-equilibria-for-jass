import tensorflow as tf


def support_to_scalar(distribution, min_value):
    """
    Transform a categorical representation to a scalar
    Calculate expectation over possible values
    """

    assert len(distribution.shape) == 2, "distribution must be batched"

    tf.assert_less(tf.abs(tf.reduce_sum(distribution, axis=1) - 1), 1e-5)

    support_size = distribution.shape[1]
    indices = tf.range(start=min_value, limit=min_value + support_size, delta=1, dtype=tf.float32)

    expected_values = tf.reduce_sum(indices * distribution, axis=1)

    return expected_values


def support_to_scalar_per_player(distribution, min_value, nr_players):
    return tf.reshape(
        support_to_scalar(tf.reshape(distribution, (-1, distribution.shape[-1])), min_value=min_value),
        (-1, nr_players))

def scalar_to_support(scalar, support_size, min_value):
    """
    Transform a scalar to a categorical representation
    Only discrete values are assumed
    """

    assert len(scalar.shape) == 2, "scalar must be batched"

    tf.debugging.assert_integer(scalar)

    scalar = tf.clip_by_value(tf.cast(scalar, tf.int32), clip_value_min=min_value, clip_value_max=min_value + support_size)

    distribution = tf.one_hot(scalar, depth=support_size)

    return distribution