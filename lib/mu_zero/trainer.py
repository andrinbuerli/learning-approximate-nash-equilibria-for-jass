import gc
import logging
import pickle
import time
from pathlib import Path

import tensorflow as tf
import tensorflow_addons as tfa
import wandb

from lib.log.base_logger import BaseLogger
from lib.metrics.metrics_manager import MetricsManager
from lib.mu_zero.network.network_base import AbstractNetwork
from lib.mu_zero.network.support_conversion import scalar_to_support, support_to_scalar, support_to_scalar_per_player
from lib.mu_zero.replay_buffer.replay_buffer_from_folder import ReplayBufferFromFolder


class MuZeroTrainer:

    def __init__(
            self,
            network: AbstractNetwork,
            replay_buffer: ReplayBufferFromFolder,
            metrics_manager: MetricsManager,
            logger: BaseLogger,
            learning_rate: float,
            weight_decay: float,
            adam_beta1: float,
            adam_beta2: float,
            adam_epsilon: float,
            min_buffer_size: int,
            updates_per_step: int,
            store_model_weights_after: int,
            policy_loss_weight: float = 1.0,
            value_loss_weight: float = 1.0,
            reward_loss_weight: float = 1.0
    ):
        self.reward_loss_weight = reward_loss_weight
        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.logger = logger
        self.metrics_manager = metrics_manager
        self.store_model_weights_after = store_model_weights_after
        self.min_buffer_size = min_buffer_size
        self.updates_per_step = updates_per_step
        self.replay_buffer = replay_buffer
        self.network = network

        self.optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=adam_beta1,
            beta_2=adam_beta2,
            epsilon=adam_epsilon)

    def fit(self, iterations: int, network_path: Path):
        logging.info("Starting alpha zero training")

        while self.replay_buffer.buffer_size < self.min_buffer_size:
            logging.info(f"waiting for buffer to fill up ({self.replay_buffer.buffer_size} / {self.min_buffer_size})")
            time.sleep(5)

        size_of_last_update_cumsum = 0
        for it in range(iterations):
            start = time.time()
            buffer_size = self.replay_buffer.buffer_size
            size_of_last_update = self.replay_buffer.size_of_last_update
            size_of_last_update_cumsum += size_of_last_update
            logging.info(f"Iteration {it}, Buffer size: {buffer_size}")


            batches = self.replay_buffer.sample_from_buffer(nr_of_batches=self.updates_per_step)

            logging.info(f"starting {self.updates_per_step} mini batch updates.")

            training_infos = self.train(batches)

            if (it % self.store_model_weights_after) == 0:
                logging.info(f'Saving checkpoint for iteration {it} at {network_path}')
                self.save_latest_network(it, network_path)

            custom_metrics = self.metrics_manager.get_latest_metrics_state()

            if isinstance(self.optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                current_lr = float(self.optimizer.lr(self.optimizer.iterations).numpy())
            else:
                current_lr = float(self.optimizer.lr.numpy())

            logging.info('Logging training steps infos...')
            for training_info in training_infos:
                grad_infos = {
                    key.replace('/', '_').replace("gradients_", "gradients/"):
                        wandb.Histogram(training_info[key].numpy())
                    for key in training_info if key.__contains__("gradients/")}

                train_infos = {key: float(training_info[key].numpy())
                               for key in training_info if not key.__contains__("gradients/")}

                data = {
                    "meta/buffer_size": buffer_size,
                    "meta/learning_rate": current_lr,
                    "meta/size_of_last_update": size_of_last_update,
                    "meta/size_of_last_update_cumsum": size_of_last_update_cumsum,
                    **train_infos,
                    **grad_infos,
                    **custom_metrics
                }

                self.logger.log(data)
                logging.debug(data)

                del train_infos
                del grad_infos
                del data

            del training_infos
            del batches
            del custom_metrics
            gc.collect()

            logging.info(f"Iteration {it} done, took: {time.time() - start}s")

    def train(self, batches):
        training_infos = list()
        for states, actions, rewards, probs, outcomes in batches:
            info, absolute_reward_errors, absolute_value_errors, policy_kls, policy_ces = self.train_step(
                tf.convert_to_tensor(states),
                tf.convert_to_tensor(actions),
                tf.convert_to_tensor(rewards),
                tf.convert_to_tensor(probs),
                tf.convert_to_tensor(outcomes))

            reward_error = {
                f"absolute_reward_error_{i}_steps_ahead": x for i, x in enumerate(absolute_reward_errors)
            }

            value_error = {
                f"absolute_value_error_{i}_steps_ahead": x for i, x in enumerate(absolute_value_errors)
            }

            policy_kls = {
                f"policy_kl_{i}_steps_ahead": x for i, x in enumerate(policy_kls)
            }

            policy_ces = {
                f"policy_ce_{i}_steps_ahead": x for i, x in enumerate(policy_ces)
            }

            training_infos.append({
                **info, **reward_error, **value_error, **policy_kls, **policy_ces
            })

            del info, absolute_reward_errors, absolute_value_errors, policy_kls, policy_ces

        return training_infos


    def save_latest_network(self, it: int, network_path: Path):
        logging.info(f'Saving latest model for iteration {it} at {network_path}')
        self.network.save(network_path)

    @tf.function
    def train_step(self, states, next_actions, rewards_target, policies_target, outcomes_target):
        trajectory_length = tf.shape(states)[1]

        policy_kls = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        policy_ces = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        absolute_value_errors = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        absolute_reward_errors = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)

        rewards_target = tf.tile(rewards_target, [1, 1, 2])
        outcomes_target = tf.tile(outcomes_target, [1, 1, 2])

        with tf.GradientTape() as tape:
            initial_states = states[:, 0]
            value, reward, policy_estimate, encoded_states = self.network.initial_inference(initial_states)

            reward_support_size = tf.shape(reward)[-1]
            min_reward = -(reward_support_size - 1) // 2
            outcome_support_size = tf.shape(value)[-1]

            reward_target_distribution = scalar_to_support(rewards_target[:, 0], support_size=reward_support_size, min_value=min_reward)
            reward_loss = self.cross_entropy(reward_target_distribution, reward)

            value_target_distribution = scalar_to_support(outcomes_target[:, 0], support_size=outcome_support_size, min_value=0)
            value_loss = self.cross_entropy(value_target_distribution, value)

            policy_loss = self.cross_entropy(policies_target[:, 0], policy_estimate)

            # ---------------Logging --------------- #
            policy_target = tf.cast(self.clip_probability_dist(policies_target[:, 0]), tf.float32)
            policy_kl_divergence_per_sample = tf.reduce_sum(
                policy_target * tf.math.log(policy_target / self.clip_probability_dist(policy_estimate)), axis=1)
            policy_kls.write(0, tf.reduce_mean(policy_kl_divergence_per_sample, name="kl_mean"))
            policy_ces.write(0, tf.reduce_mean(policy_loss, name="p_loss"))

            expected_value = support_to_scalar_per_player(value, min_value=0, nr_players=4)
            absolute_value_errors.write(0, tf.reduce_mean(tf.abs(expected_value - tf.cast(outcomes_target[:, 0], tf.float32)), name="val_mae"))
            expected_reward = support_to_scalar_per_player(reward, min_value=min_reward, nr_players=4)
            absolute_reward_errors.write(0, tf.reduce_mean(tf.abs(expected_reward - tf.cast(rewards_target[:, 0], tf.float32)), name="reward_mae"))
            # ---------------Logging --------------- #


            for i in range(trajectory_length - 1):
                next_action = tf.reshape(next_actions[:, i], [-1, 1])
                value, reward, policy_estimate, encoded_states = self.network.recurrent_inference(encoded_states, next_action)

                reward_target_distribution = scalar_to_support(rewards_target[:, i+1], support_size=reward_support_size,
                                                               min_value=min_reward)
                reward_loss += self.cross_entropy(reward_target_distribution, reward)

                value_target_distribution = scalar_to_support(outcomes_target[:, i+1], support_size=outcome_support_size,
                                                              min_value=0)
                value_loss += self.cross_entropy(value_target_distribution, value)

                policy_ce = self.cross_entropy(policies_target[:, i + 1], policy_estimate)
                policy_loss += policy_ce

                # ---------------Logging --------------- #
                policy_target = tf.cast(self.clip_probability_dist(policies_target[:, 0]), tf.float32)
                policy_kl_divergence_per_sample = tf.reduce_sum(
                    policy_target * tf.math.log(policy_target / self.clip_probability_dist(policy_estimate)), axis=1)
                policy_kls.write(i+1, tf.reduce_mean(policy_kl_divergence_per_sample, name="kl_mean"))
                policy_ces.write(i+1, tf.reduce_mean(policy_ce, name="ce_mean"))

                expected_value = support_to_scalar_per_player(value, min_value=0, nr_players=4)
                absolute_value_errors.write(i+1, tf.reduce_mean(tf.abs(expected_value - tf.cast(outcomes_target[:, i+1], tf.float32)), name="val_mae"))
                expected_reward = support_to_scalar_per_player(reward, min_value=min_reward, nr_players=4)
                absolute_reward_errors.write(i+1, tf.reduce_mean(tf.abs(expected_reward - tf.cast(rewards_target[:, i+1], tf.float32)), name="reard_mae"))
                # ---------------Logging --------------- #

            loss = tf.reduce_mean(
                self.reward_loss_weight * tf.reduce_mean(reward_loss, axis=-1, name="rewards_loss") +
                self.value_loss_weight * tf.reduce_mean(value_loss, axis=-1, name="value_loss") +
                self.policy_loss_weight * policy_loss, name="loss_mean")

        gradients = tape.gradient(loss, self.network.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        gradient_hists = {f"gradients/layer_{i}_{x.name}": g
                          for i, (g, x) in enumerate(zip(gradients, self.network.trainable_variables))}

        # inspired by https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
        squared_weights_sum = tf.reduce_sum([tf.reduce_sum(x ** 2) for x in self.network.trainable_weights])

        abs_reward_errors = absolute_reward_errors.stack()
        abs_value_errors = absolute_value_errors.stack()
        pl_kls = policy_kls.stack()
        pl_ces = policy_ces.stack()

        absolute_reward_errors.close(), absolute_value_errors.close()
        policy_kls.close(), policy_ces.close()

        return {
            "training/reward_loss": tf.reduce_mean(reward_loss),
            "training/value_loss": tf.reduce_mean(value_loss),
            "training/policy_loss": tf.reduce_mean(policy_loss),
            "training/squared_weights_sum": squared_weights_sum,
            "training/loss": loss,
            **gradient_hists
        }, abs_reward_errors, abs_value_errors, pl_kls, pl_ces


    def cross_entropy(self, target, estimate):
        target = tf.cast(target, dtype=tf.float32)
        # clipping of output for ce is important, if not done, will result in exploding gradients
        estimate = self.clip_probability_dist(estimate)
        cross_entropy = -tf.reduce_sum(target * tf.math.log(estimate), axis=-1)

        return cross_entropy

    @staticmethod
    def clip_probability_dist(dist, eps=1e-07):
        return tf.clip_by_value(dist, eps, 1. - eps)
