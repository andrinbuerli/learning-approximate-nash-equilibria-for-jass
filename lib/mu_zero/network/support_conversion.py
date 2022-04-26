import math

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

def scalar_to_support(scalar_m, support_size, min_value, dldl=False):
    """
    Transform a scalar to a categorical representation
    Only discrete values are assumed
    """

    assert len(scalar_m.shape) == 2, "scalar must be batched"

    tf.debugging.assert_integer(scalar_m)

    if dldl:
        pi = tf.constant(math.pi, dtype=tf.float32)
        rng = tf.range(support_size, dtype=tf.float32)
        rng = tf.tile(rng[None, None, :], (1, 4, 1))

        points_left_in_game = tf.reduce_sum(scalar_m, axis=1) // 2
        sigma = tf.maximum(tf.cast(points_left_in_game / 4, tf.float32), 1)[:, None, None]
        norm_factor = 1 / (tf.math.sqrt(2 * pi) * sigma)
        scalar_m = tf.cast(scalar_m[:, :, None], tf.float32)
        distribution = norm_factor * tf.exp(-(rng - scalar_m) ** 2 / (2 * sigma ** 2))

        distribution /= tf.reduce_sum(distribution, axis=-1, keepdims=True)
    else:
        scalar_m = tf.clip_by_value(tf.cast(scalar_m, tf.int32), clip_value_min=min_value,
                                    clip_value_max=min_value + (support_size - 1))

        distribution_m = tf.one_hot(scalar_m, depth=support_size)
        distribution = distribution_m

    return distribution