from pathlib import Path

import numpy as np
import tensorflow as tf
from jass.features.feature_example_buffer import parse_feature_example
from jass.features.features_conv_full import FeaturesSetConvFull

from lib.environment.networking.worker_config import WorkerConfig
from lib.metrics.base_async_metric import BaseAsyncMetric
from lib.mu_zero.network.network_base import AbstractNetwork
from lib.mu_zero.network.support_conversion import support_to_scalar




def _calculate_vpkl_(current_positions, policy_estimate, x):
    current_states = tf.reshape(tf.gather_nd(x, current_positions), (-1,) + FeaturesSetConvFull.FEATURE_SHAPE)
    valid_cards = tf.reshape(current_states[:, :, :, FeaturesSetConvFull.CH_CARDS_VALID], [-1, 36])
    trump_valid = tf.tile(
        tf.reshape(tf.reduce_max(current_states[:, :, :, FeaturesSetConvFull.CH_TRUMP_VALID], axis=(1, 2)), [-1, 1]),
        [1, 6])
    push_valid = tf.reshape(tf.reduce_max(current_states[:, :, :, FeaturesSetConvFull.CH_PUSH_VALID], axis=(1, 2)),
                            [-1, 1])
    valid_actions = tf.concat([
        valid_cards,
        trump_valid,
        push_valid
    ], axis=-1)

    valid_actions = valid_actions / tf.reduce_sum(valid_actions)

    policy_estimate = tf.clip_by_value(policy_estimate, 1e-7, 1. - 1e-7)
    valid_actions = tf.clip_by_value(valid_actions, 1e-7, 1. - 1e-7)
    kl = tf.reduce_mean(
        tf.reduce_sum(valid_actions * tf.math.log(valid_actions / policy_estimate), axis=1)).numpy()
    return kl


def _calculate_batched_vpkl_(network: AbstractNetwork, iterator, n_steps_ahead, f_shape, l_shape):
    x, y = next(iterator)

    x = tf.reshape(x, (-1,) + f_shape)
    y = tf.reshape(y, (-1,) + l_shape)

    batch_size = x.shape[0]
    trajectory_length = x.shape[1]
    position = np.random.choice(range(trajectory_length))
    positions = np.array(list(zip(range(batch_size), np.repeat(position, batch_size))))

    initial_states = tf.gather_nd(x, positions)
    value, reward, policy_estimate, encoded_states = network.initial_inference(initial_states)

    min_tensor = tf.stack((tf.range(batch_size), tf.repeat(trajectory_length - 1, batch_size)), axis=1)
    zeros = tf.zeros(batch_size, dtype=tf.int32)
    current_positions = positions

    kls = []
    kl = _calculate_vpkl_(current_positions, policy_estimate, x)
    kls.append(float(kl))

    for i in range(n_steps_ahead):
        supervised_policy = tf.gather_nd(y, current_positions)[:, :43]
        assert all(tf.reduce_max(supervised_policy, axis=-1) == 1)

        actions = tf.reshape(tf.argmax(supervised_policy, axis=-1), [-1, 1])
        value, reward, policy_estimate, encoded_states =  network.recurrent_inference(encoded_states, actions)

        current_positions = tf.minimum(positions + [0, (i + 1)], min_tensor)  # repeat last action at end
        supervised_policy = tf.gather_nd(y, current_positions)[:, :43]
        # solve if trajectory hans only length of 37
        current_positions = current_positions - tf.stack((zeros, tf.cast(tf.reduce_sum(supervised_policy, axis=-1) == 0, tf.int32)), axis=1)

        kl = _calculate_vpkl_(current_positions, policy_estimate, x)
        kls.append(float(kl))

    return {
        f"vpkl_{i}_steps_ahead": x for i, x in enumerate(kls)
    }


class VPKL(BaseAsyncMetric):

    def get_params(self, thread_nr: int, network: AbstractNetwork, init_vars=None) -> []:
        iterator = init_vars
        return network, iterator, self.n_steps_ahead, self.trajectory_feature_shape, self.trajectory_label_shape

    def init_dataset(self):
        ds = tf.data.TFRecordDataset(self.tf_record_files)
        ds = ds.map(lambda x: parse_feature_example(x,
                          feature_length=self.trajectory_length*self.feature_length,
                          label_length=self.trajectory_length*self.label_length))
        ds = ds.batch(self.samples_per_calculation).repeat()
        return iter(ds)

    def __init__(
            self,
            samples_per_calculation: int,
            feature_length: int,
            feature_shape: (int, int, int),
            label_length: int,
            worker_config: WorkerConfig,
            network_path: str,
            n_steps_ahead: int,
            trajectory_length: int = 38,
            tf_record_files: [str] = None):

        self.trajectory_length = trajectory_length
        if tf_record_files is None:
            tf_record_files = [str(x.resolve()) for x in
                               (Path(__file__).parent.parent.parent / "resources" / "supervised_data").glob("*.tfrecord")]

        self.n_steps_ahead = n_steps_ahead
        self.samples_per_calculation = samples_per_calculation
        self.feature_length = feature_length
        self.feature_shape = feature_shape
        self.label_length = label_length
        self.tf_record_files = tf_record_files

        self.trajectory_feature_shape = (self.trajectory_length, feature_length)
        self.trajectory_label_shape = (self.trajectory_length, label_length)

        super().__init__(worker_config, network_path, parallel_threads=1,
                         metric_method=_calculate_batched_vpkl_, init_method=self.init_dataset)

    def get_name(self, i: int):
        return f"vpkl_{i}_steps_ahead"