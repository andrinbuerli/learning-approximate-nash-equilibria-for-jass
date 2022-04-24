import tensorflow as tf

def support_to_scalar(distribution, min_value):
    """
    Transform a categorical representation to a scalar
    Calculate expectation over possible values
    """

    distribution = tf.convert_to_tensor(distribution, dtype=tf.float32)

    assert len(distribution.shape) == 2, f"distribution must be batched, {distribution.shape}"

    tf.assert_less(tf.abs(tf.reduce_sum(distribution, axis=1) - 1), 1e-5)

    support_size = distribution.shape[1]
    indices = tf.range(start=min_value, limit=min_value + support_size, delta=1, dtype=tf.float32)

    expected_values = tf.reduce_sum(indices * distribution, axis=1)
    # expected_values = tf.argmax(distribution, axis=-1)

    return expected_values


def support_to_scalar_per_player(distribution, min_value, nr_players):
    return tf.reshape(
        support_to_scalar(tf.reshape(distribution, (-1, distribution.shape[-1])), min_value=min_value),
        (-1, nr_players))

def scalar_to_support(scalar_m, support_size, min_value, augment=False):
    """
    Transform a scalar to a categorical representation
    Only discrete values are assumed
    """

    assert len(scalar_m.shape) == 2, "scalar must be batched"

    tf.debugging.assert_integer(scalar_m)

    scalar_m = tf.clip_by_value(tf.cast(scalar_m, tf.int32), clip_value_min=min_value,
                                clip_value_max=min_value + (support_size - 1))
    scalar_l = tf.clip_by_value(tf.cast(scalar_m - 1, tf.int32), clip_value_min=min_value,
                                clip_value_max=min_value + (support_size - 1))
    scalar_h = tf.clip_by_value(tf.cast(scalar_m + 1, tf.int32), clip_value_min=min_value,
                                clip_value_max=min_value + (support_size - 1))

    distribution_m = tf.one_hot(scalar_m, depth=support_size)
    distribution_l = tf.one_hot(scalar_l, depth=support_size)
    distribution_h = tf.one_hot(scalar_h, depth=support_size)

    # make support non-one hot!
    shape = tf.shape(distribution_m)
    if augment:
        rand = tf.random.uniform((shape[0], shape[1], 1), minval=0, maxval=0.4)
    else:
        rand = 0.0
    distribution = (rand / 2) * distribution_l + (1 - rand) * distribution_m + (rand / 2) * distribution_h

    return distribution