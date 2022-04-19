from pathlib import Path

import numpy as np
import tensorflow as tf
from jass.features.feature_example_buffer import parse_feature_example
from matplotlib import pyplot as plt

from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.features.features_cpp_conv_cheating import FeaturesSetCppConvCheating
from lib.metrics.base_async_metric import BaseAsyncMetric
from lib.mu_zero.network.network_base import AbstractNetwork
from lib.mu_zero.network.support_conversion import support_to_scalar

def _make_plots_(network: AbstractNetwork, states, y, f_shape, l_shape, features):
    states = tf.reshape(states, f_shape)
    y = tf.reshape(y, l_shape)

    cum_rewards = [np.array([0, 0])]

    all_rewards = [np.array([0, 0])]
    all_reward_estimates = [np.array([0, 0])]

    players = []
    kls = []
    values = []

    value, reward, policy, encoded_state = network.initial_inference(states[0][None])

    prev_points = [0, 0]
    for i in range(37):
        current_state = tf.reshape(states[i], features.FEATURE_SHAPE)

        valid_cards = tf.reshape(current_state[:, :, features.CH_CARDS_VALID], [36])
        trump_valid = tf.reduce_max(current_state[:, :, features.CH_TRUMP_VALID])
        push_valid = tf.reduce_max(current_state[:, :, features.CH_PUSH_VALID])
        valid_actions = tf.concat([
            valid_cards,
            np.repeat(trump_valid, 6),
            [push_valid]
        ], axis=-1).numpy()
        kl = (valid_actions * np.log(valid_actions / policy[0].numpy() + 1e-7)).sum()
        kls.append(kl)
        values.append(support_to_scalar(value[0], min_value=0).numpy())

        current_player = tf.math.argmax(current_state[0, 0, features.CH_PLAYER:features.CH_PLAYER + 4])
        players.append(current_player)
        current_team = current_player % 2
        current_points = tf.cast(current_state[0, 0, features.CH_POINTS_OWN:(features.CH_POINTS_OPP + 1)] * 157, tf.int32)

        if current_team == 0:
            current_points = current_points
        else:
            current_points = current_points[::-1]

        rewards = current_points - prev_points

        cum_rewards.append(cum_rewards[-1] + rewards)
        all_rewards.append(rewards)

        action = tf.math.argmax(y[i, :43])
        value, reward, policy, encoded_state = network.recurrent_inference(encoded_state, [[action]])
        all_reward_estimates.append(support_to_scalar(reward[0], min_value=0).numpy())

    fig_kls = plt.figure()
    plt.plot(kls)
    colors = ["red", "blue", "green", "violet"]
    legend = [True, True, True, True]
    for i, (kl, player) in enumerate(zip(kls, players)):
        plt.scatter(i, kl, c=colors[player], label=f"player {player}" if legend[player] else None)
        legend[player] = False
    plt.legend()

    fig_value = plt.figure()
    plt.plot(np.array(values)[:, 0], marker="x", label="pred 0", alpha=0.5, color="green")
    plt.plot(np.array(values)[:, 1], marker="x", label="pred 1", alpha=0.5, color="red")

    outcome = y[0, 45:]
    plt.plot(outcome[0] - np.array(cum_rewards)[:, 0], marker="o", label="value 0", alpha=0.5, color="green")
    plt.plot(outcome[1] - np.array(cum_rewards)[:, 1], marker="o", label="value 1", alpha=0.5, color="red")

    plt.legend()
    plt.xlabel("Moves")
    plt.ylabel("Cumulative Reward")

    fig_reward = plt.figure()
    legend = True
    for i, (r, r_estimate) in enumerate(zip(all_rewards, all_reward_estimates)):
        plt.scatter(i, r[0], marker="o", color="green", alpha=0.5, label="reward team 0" if legend else None)
        plt.scatter(i, r[1], marker="o", color="red", alpha=0.5, label="reward team 1" if legend else None)
        plt.scatter(i, r_estimate[0], marker="x", color="green", label="reward estimate team 0" if legend else None)
        plt.scatter(i, r_estimate[1], marker="x", color="red", label="reward estimate team 1" if legend else None)
        legend = False

    plt.xlabel("Moves")
    plt.ylabel("Reward")
    plt.legend(bbox_to_anchor=(1.1, 1))

    return {
        f"Visualizations/reward": fig_reward,
        f"Visualizations/value": fig_value,
        f"Visualizations/policy": fig_kls
    }


class GameVisualisation(BaseAsyncMetric):

    def get_params(self, thread_nr: int, network: AbstractNetwork, init_vars=None) -> []:
        x, y = init_vars
        return network, x, y,self.trajectory_feature_shape, \
               self.trajectory_label_shape, self.worker_config.network.feature_extractor

    def init_dataset(self):
        ds = tf.data.TFRecordDataset(self.tf_record_files)
        ds = ds.map(lambda x: parse_feature_example(x,
                          feature_length=self.trajectory_length*self.feature_length,
                          label_length=self.trajectory_length*self.label_length))
        x, y = next(iter(ds))
        return x, y

    def __init__(
            self,
            label_length: int,
            worker_config: WorkerConfig,
            network_path: str,
            trajectory_length: int = 38,
            tf_record_files: [str] = None):

        cheating_mode = type(worker_config.network.feature_extractor) == FeaturesSetCppConvCheating

        file_ending = "*.perfect.tfrecord" if cheating_mode else "*.imperfect.tfrecord"
        self.trajectory_length = trajectory_length
        if tf_record_files is None:
            tf_record_files = [str(x.resolve()) for x in
                               (Path(__file__).parent.parent.parent / "resources" / "supervised_data").glob(file_ending)]

        self.feature_length = worker_config.network.feature_extractor.FEATURE_LENGTH
        self.feature_shape = worker_config.network.feature_extractor.FEATURE_SHAPE
        self.label_length = label_length
        self.tf_record_files = tf_record_files

        self.trajectory_feature_shape = (self.trajectory_length, self.feature_length)
        self.trajectory_label_shape = (self.trajectory_length, label_length)

        super().__init__(worker_config, network_path, parallel_threads=1,
                         metric_method=_make_plots_, init_method=self.init_dataset)

    def get_latest_result(self) -> dict:
        while self.result_queue.qsize() > 0:
            self._latest_result = self.result_queue.get()
            return self._latest_result
        return {
            f"Visualizations/reward": None,
            f"Visualizations/value": None,
            f"Visualizations/policy": None
        }

    def get_name(self):
        return f"save"