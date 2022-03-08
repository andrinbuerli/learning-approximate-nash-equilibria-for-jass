from pathlib import Path

import numpy as np
import tensorflow as tf
from jass.features.feature_example_buffer import parse_feature_example
from jass.features.features_conv_full import FeaturesSetConvFull

from lib.environment.networking.worker_config import WorkerConfig
from lib.metrics.base_async_metric import BaseAsyncMetric
from lib.mu_zero.network.network_base import AbstractNetwork
from lib.mu_zero.network.support_conversion import support_to_scalar



def _calculate_mae_(outcomes, value):
    value_pred = tf.reshape(support_to_scalar(tf.reshape(value, (-1, value.shape[-1])), min_value=0), (-1, 4))
    mae = tf.reduce_mean(tf.abs(value_pred - tf.cast(outcomes, tf.float32)))
    return mae

def _calculate_batched_save_(network: AbstractNetwork, iterator, n_steps_ahead, f_shape, l_shape):
    x, y = next(iterator)

    x = tf.reshape(x, (-1,) + f_shape)
    y = tf.reshape(y, (-1,) + l_shape)

    batch_size = x.shape[0]
    trajectory_length = 37
    position = np.random.choice(range(trajectory_length - n_steps_ahead))
    positions = np.array(list(zip(range(batch_size), np.repeat(position, batch_size))))

    initial_positions = tf.gather_nd(x, positions)
    value, reward, policy_estimate, encoded_states = network.initial_inference(initial_positions)

    reshaped = tf.reshape(initial_positions, (-1,) + FeaturesSetConvFull.FEATURE_SHAPE)
    current_player = tf.reduce_max(reshaped[:, :, :, FeaturesSetConvFull.CH_PLAYER:FeaturesSetConvFull.CH_PLAYER + 4], axis=(1,2))
    current_team = tf.argmax(current_player, axis=-1) % 2
    outcomes = tf.cast(tf.gather_nd(y, positions)[:, 43:45] * 157, tf.int32)

    current_teams = 1 - tf.cast((
            tf.tile([[0, 1, 0, 1]], [batch_size, 1]) == tf.cast(tf.reshape(tf.repeat(current_team, 4), [-1, 4]), tf.int32)),
        tf.int32)
    outcomes = tf.gather_nd(outcomes, tf.stack((tf.reshape(tf.repeat(tf.range(batch_size), 4), [-1, 4]), current_teams), axis=-1))

    assert all(tf.reduce_sum(outcomes, axis=-1) == 157 * 2), f"{tf.reduce_sum(outcomes, axis=-1)}, should match 157 * 2"

    min_tensor = tf.stack((tf.range(batch_size), tf.repeat(trajectory_length - 1, batch_size)), axis=1)
    zeros = tf.zeros(batch_size, dtype=tf.int32)
    current_positions = positions
    maes = []
    mae = _calculate_mae_(outcomes, value)
    maes.append(float(mae))
    for i in range(n_steps_ahead):
        supervised_policy = tf.gather_nd(y, current_positions)[:, :43]
        assert all(tf.reduce_max(supervised_policy, axis=-1) == 1)

        actions = tf.reshape(tf.argmax(supervised_policy, axis=-1), [-1, 1])
        value, reward, policy_estimate, encoded_states =  network.recurrent_inference(encoded_states, actions)

        current_positions = tf.minimum(positions + [0, (i + 1)], min_tensor)  # repeat last action at end

        supervised_policy = tf.gather_nd(y, current_positions)[:, :43]
        # solve if trajectory hans only length of 37
        current_positions = current_positions - tf.stack((zeros, tf.cast(tf.reduce_sum(supervised_policy, axis=-1) == 0, tf.int32)), axis=1)

        current_teams = (current_teams + 1) % 2
        outcomes = tf.gather_nd(
            outcomes, tf.stack((tf.reshape(tf.repeat(tf.range(batch_size), 4), [-1, 4]), current_teams), axis=-1))

        assert all(
            tf.reduce_sum(outcomes, axis=-1) == 157 * 2), f"{tf.reduce_sum(outcomes, axis=-1)}, should match 157 * 2"

        mae = _calculate_mae_(outcomes, value)
        maes.append(float(mae))

    return {
        f"SAVE/save_{i}_steps_ahead": x for i, x in enumerate(maes)
    }


class SAVE(BaseAsyncMetric):

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
                         metric_method=_calculate_batched_save_, init_method=self.init_dataset)

    def get_name(self):
        return f"save"