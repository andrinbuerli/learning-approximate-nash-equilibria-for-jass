import gc
from pathlib import Path

import numpy as np
import tensorflow as tf
import wandb
from jass.features.feature_example_buffer import parse_feature_example
from matplotlib import pyplot as plt

from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.features.features_cpp_conv_cheating import FeaturesSetCppConvCheating
from lib.metrics.base_async_metric import BaseAsyncMetric
from lib.mu_zero.network.network_base import AbstractNetwork
from lib.mu_zero.network.support_conversion import support_to_scalar


def clear_all_matplotlib_cache():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def _make_plots_(network: AbstractNetwork, iterator, f_shape, l_shape, features, mdp_value):
    clear_all_matplotlib_cache()

    states, y = next(iterator)
    states = tf.reshape(states, f_shape)
    y = tf.reshape(y, l_shape)

    cum_rewards = [np.array([0, 0])]

    all_rewards = []
    all_reward_estimates = []

    players = []
    policy_kls = []
    player_kls = []
    hand_kls = []
    values = []

    idx = np.random.choice(range(38), 1)[0]
    value, reward, policy, player, hand, is_terminal, encoded_state = network.initial_inference(states[idx][None], all_preds=True)

    prev_points = [0, 0]
    i = idx
    while i < 38 and y[i, :43].numpy().max() > 0:
        current_state = tf.reshape(states[i], features.FEATURE_SHAPE)

        current_player = tf.math.argmax(current_state[0, 0, features.CH_PLAYER:features.CH_PLAYER + 4])

        valid_cards = tf.reshape(current_state[:, :, features.CH_CARDS_VALID], [36])
        trump_valid = tf.reduce_max(current_state[:, :, features.CH_TRUMP_VALID])
        push_valid = tf.reduce_max(current_state[:, :, features.CH_PUSH_VALID])
        valid_actions = tf.concat([
            valid_cards,
            np.repeat(trump_valid, 6),
            [push_valid]
        ], axis=-1).numpy()
        policy_kl = (valid_actions * np.log(valid_actions / policy[0].numpy() + 1e-7)).sum()
        policy_kls.append(policy_kl)

        player_kl = np.log(1 / player[0][current_player].numpy() + 1e-7)
        player_kls.append(player_kl)

        if hasattr(features, "CH_HAND"):
            true_hand = current_state[:, :, features.CH_HAND].numpy().reshape(-1)
        else:
            true_hand = current_state[:, :, features.CH_HANDS+current_player].numpy().reshape(-1)

        true_hand_dist = true_hand / true_hand.sum()
        pred_hand_dist = hand[0].numpy() / hand[0].numpy().sum()
        hand_kl = (true_hand_dist * np.log(true_hand_dist / pred_hand_dist + 1e-7)).sum()
        hand_kls.append(hand_kl)

        support_size = tf.shape(value)[-1]
        values.append(support_to_scalar(value[0], min_value=-support_size//2).numpy())

        players.append(current_player)
        current_team = current_player % 2
        current_points = tf.cast(current_state[0, 0, features.CH_POINTS_OWN:(features.CH_POINTS_OPP + 1)] * 157, tf.int32)

        if current_team == 0:
            current_points = current_points
        else:
            current_points = current_points[::-1]

        rewards = current_points - prev_points
        prev_points = current_points

        cum_rewards.append(cum_rewards[-1] + rewards)
        all_rewards.append(rewards)

        action = tf.math.argmax(y[i, :43])
        support_size = tf.shape(reward)[-1]
        all_reward_estimates.append(support_to_scalar(reward[0], min_value=-support_size//2).numpy())
        value, reward, policy, player, hand, is_terminal, encoded_state = network.recurrent_inference(encoded_state, [[action]], all_preds=True)
        i += 1

    fig_policy_kls = plt.figure()
    plt.plot(policy_kls)
    colors = ["red", "blue", "green", "violet"]
    legend = [True, True, True, True]
    for i, (policy_kl, player) in enumerate(zip(policy_kls, players)):
        plt.scatter(i, policy_kl, c=colors[player], label=f"player {player}" if legend[player] else None)
        legend[player] = False
    plt.legend()
    policy_img = wandb.Image(fig_policy_kls)
    del fig_policy_kls
    clear_all_matplotlib_cache()

    fig_player_kls = plt.figure()
    plt.plot(player_kls)
    legend = [True, True, True, True]
    for i, (player_kl, player) in enumerate(zip(player_kls, players)):
        plt.scatter(i, player_kl, c=colors[player], label=f"player {player}" if legend[player] else None)
        legend[player] = False
    plt.legend()
    player_img = wandb.Image(fig_player_kls)
    del fig_player_kls
    clear_all_matplotlib_cache()

    fig_hand_kls = plt.figure()
    plt.plot(hand_kls)
    legend = [True, True, True, True]
    for i, (hand_kl, player) in enumerate(zip(hand_kls, players)):
        plt.scatter(i, hand_kl, c=colors[player], label=f"player {player}" if legend[player] else None)
        legend[player] = False
    plt.legend()
    hand_img = wandb.Image(fig_hand_kls)
    del fig_hand_kls
    clear_all_matplotlib_cache()

    fig_value = plt.figure()
    values = np.array(values)
    plt.plot(values[:, 0], marker="x", label="pred 0", alpha=0.5, color="green")
    plt.plot(values[:, 1], marker="x", label="pred 1", alpha=0.5, color="red")

    beginning_team = players[0] % 2

    if beginning_team == 0:
        outcome = (y[0, 43:45] * 157).numpy().astype(int)
    else:
        outcome = (y[0, 43:45] * 157).numpy().astype(int)[::-1]

    if mdp_value:
        plt.plot(outcome[0] - np.array(cum_rewards)[1:, 0], marker="o", label="value 0", alpha=0.5, color="green")
        plt.plot(outcome[1] - np.array(cum_rewards)[1:, 1], marker="o", label="value 1", alpha=0.5, color="red")
    else:
        plt.plot(np.repeat(outcome[0], values.shape[0]), marker="o", label="value 0", alpha=0.5, color="green")
        plt.plot(np.repeat(outcome[1], values.shape[0]), marker="o", label="value 1", alpha=0.5, color="red")

    plt.legend()
    plt.xlabel("Moves")
    plt.ylabel("Cumulative Reward")
    value_img = wandb.Image(fig_value)
    del fig_value
    clear_all_matplotlib_cache()

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
    reward_img = wandb.Image(fig_reward)
    del fig_reward
    clear_all_matplotlib_cache()

    plt.show()
    gc.collect()

    return {
        f"Visualizations/reward": reward_img,
        f"Visualizations/value": value_img,
        f"Visualizations/policy": policy_img,
        f"Visualizations/player": player_img,
        f"Visualizations/hand": hand_img
    }


class GameVisualisation(BaseAsyncMetric):

    def get_params(self, thread_nr: int, network: AbstractNetwork, init_vars=None) -> []:
        interator = init_vars
        return network, interator,self.trajectory_feature_shape, \
               self.trajectory_label_shape, self.worker_config.network.feature_extractor, self.mdp_value

    def init_dataset(self):
        ds = tf.data.TFRecordDataset(self.tf_record_files)
        ds = ds.map(lambda x: parse_feature_example(x,
                          feature_length=self.trajectory_length*self.feature_length,
                          label_length=self.trajectory_length*self.label_length)).repeat()
        return iter(ds)

    def __init__(
            self,
            label_length: int,
            worker_config: WorkerConfig,
            network_path: str,
            mdp_value: bool,
            trajectory_length: int = 38,
            tf_record_files: [str] = None):

        self.mdp_value = mdp_value
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

    def get_name(self):
        return f"save"