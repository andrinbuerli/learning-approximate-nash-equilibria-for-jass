from pathlib import Path

import numpy as np
import tensorflow as tf
from jass.features.feature_example_buffer import parse_feature_example

from lib.environment.networking.worker_config import WorkerConfig
from lib.metrics.base_async_metric import BaseAsyncMetric
from lib.mu_zero.network.network_base import AbstractNetwork


def _calculate_batched_spkl_(network: AbstractNetwork, iterator, n_steps_ahead, f_shape, l_shape):
    x, y = next(iterator)

    x = tf.reshape(x, (-1,) + f_shape)
    y = tf.reshape(y, (-1,) + l_shape)

    batch_size = x.shape[0]
    trajectory_length = x.shape[1]
    position = np.random.choice(range(trajectory_length))
    positions = np.array(list(zip(range(batch_size), np.repeat(position, batch_size))))

    value, reward, policy_estimate, encoded_states = network.initial_inference(tf.gather_nd(x, positions))

    supervised_policy = tf.gather_nd(y, positions)[:, :43]
    assert all(tf.reduce_max(supervised_policy, axis=-1) == 1)

    min_tensor = tf.stack((tf.range(batch_size), tf.repeat(trajectory_length - 1, batch_size)), axis=1)
    zeros = tf.zeros(batch_size, dtype=tf.int32)
    current_positions = positions
    kls = []
    kl = _calculate_spkl_(policy_estimate, supervised_policy)
    kls.append(float(kl))

    for i in range(n_steps_ahead):
        actions = tf.reshape(tf.argmax(supervised_policy, axis=-1), [-1, 1])
        value, reward, policy_estimate, encoded_states =  network.recurrent_inference(encoded_states, actions)

        current_positions = tf.minimum(positions + [0, (i + 1)], min_tensor)  # repeat last action at end

        supervised_policy = tf.gather_nd(y, current_positions)[:, :43]
        # solve if trajectory hans only length of 37
        current_positions = current_positions - tf.stack((zeros, tf.cast(tf.reduce_sum(supervised_policy, axis=-1) == 0, tf.int32)), axis=1)

        supervised_policy = tf.gather_nd(y, current_positions)[:, :43]
        assert all(tf.reduce_max(supervised_policy, axis=-1) == 1)

        kl = _calculate_spkl_(policy_estimate, supervised_policy)
        kls.append(float(kl))

    return {
        f"spkl_{i}_steps_ahead": x for i, x in enumerate(kls)
    }


def _calculate_spkl_(policy_estimate, supervised_policy):
    policy_estimate = tf.clip_by_value(policy_estimate, 1e-7, 1. - 1e-7)
    supervised_policy = tf.clip_by_value(supervised_policy, 1e-7, 1. - 1e-7)
    kl = tf.reduce_mean(
        tf.reduce_sum(supervised_policy * tf.math.log(supervised_policy / policy_estimate), axis=1)).numpy()
    return kl


class SPKL(BaseAsyncMetric):

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
                         metric_method=_calculate_batched_spkl_, init_method=self.init_dataset)

    def get_name(self):
        return f"spkl"