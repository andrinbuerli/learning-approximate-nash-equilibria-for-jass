import gc
import logging
import time
from pathlib import Path

import tensorflow as tf
import tensorflow_addons as tfa
import wandb

from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.log.base_logger import BaseLogger
from lib.metrics.metrics_manager import MetricsManager
from lib.mu_zero.network.network_base import AbstractNetwork
from lib.mu_zero.network.support_conversion import scalar_to_support, support_to_scalar_per_player
from lib.mu_zero.replay_buffer.replay_buffer_from_folder import ReplayBufferFromFolder


class MuZeroTrainer:

    def __init__(
            self,
            network: AbstractNetwork,
            replay_buffer: ReplayBufferFromFolder,
            metrics_manager: MetricsManager,
            config: WorkerConfig,
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
            reward_loss_weight: float = 1.0,
            store_weights:bool = True,
            store_buffer:bool = False
    ):
        self.config = config
        self.store_buffer = store_buffer
        self.store_weights = store_weights
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

        channel_mapping = {k: v for k, v in dict(vars(FeaturesSetCppConv)).items() if isinstance(v, int) and v < 100}
        self.feature_names = {}
        for c in range(FeaturesSetCppConv.FEATURE_SHAPE[-1]):
            for k, v in channel_mapping.items():
                if v > c:
                    break
                self.feature_names[c] = k

    def fit(self, iterations: int, network_path: Path):
        logging.info("Starting alpha zero training")

        while self.replay_buffer.buffer_size < self.min_buffer_size:
            logging.info(f"waiting for buffer to fill up ({self.replay_buffer.buffer_size} / {self.min_buffer_size})")
            time.sleep(5)

            if self.store_buffer:
                self.replay_buffer.save()

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

                if self.store_buffer:
                    self.replay_buffer.save()

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
            info, absolute_reward_errors, absolute_value_errors, policy_kls, policy_ces, ls_entropies, ft = self.train_step(
                tf.convert_to_tensor(states),
                tf.convert_to_tensor(actions),
                tf.convert_to_tensor(rewards),
                tf.convert_to_tensor(probs),
                tf.convert_to_tensor(outcomes))

            reward_error = {
                f"ARE/absolute_reward_error_{i}_steps_ahead": x for i, x in enumerate(absolute_reward_errors)
            }

            value_error = {
                f"AVE/absolute_value_error_{i}_steps_ahead": x for i, x in enumerate(absolute_value_errors)
            }

            policy_kls = {
                f"PKL/policy_kl_{i}_steps_ahead": x for i, x in enumerate(policy_kls)
            }

            policy_ces = {
                f"PCE/policy_ce_{i}_steps_ahead": x for i, x in enumerate(policy_ces)
            }

            ls_entropies = {
                f"LSE/latent_space_entropy_{i}_steps_ahead": x for i, x in enumerate(ls_entropies)
            }

            train_input_dict = {
                f"train_input/channel_{c}_{self.feature_names[c]}": ft[c] for c in range(FeaturesSetCppConv.FEATURE_SHAPE[-1])
            }

            training_infos.append({
                **info, **reward_error, **value_error, **policy_kls, **policy_ces, **ls_entropies, **train_input_dict
            })

            del info, absolute_reward_errors, absolute_value_errors, policy_kls, policy_ces

        return training_infos


    def save_latest_network(self, it: int, network_path: Path):
        if self.store_weights:
            logging.info(f'Saving latest model for iteration {it} at {network_path}')
            self.network.save(network_path)

    @tf.function
    def train_step(self, states, next_actions, rewards_target, policies_target, outcomes_target):
        batch_size = tf.shape(states)[0]
        trajectory_length = tf.shape(states)[1]

        policy_kls = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        policy_ces = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        absolute_value_errors = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        absolute_reward_errors = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        latent_space_entropy = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)

        rewards_target = tf.tile(rewards_target, [1, 1, 2])
        outcomes_target = tf.tile(outcomes_target, [1, 1, 2])

        with tf.GradientTape() as tape:
            initial_states = states[:, 0]
            value, reward, policy_estimate, encoded_states = self.network.initial_inference(
                initial_states, training=True)

            reward_support_size = tf.shape(reward)[-1]
            outcome_support_size = tf.shape(value)[-1]

            reward_loss = tf.zeros((batch_size, 4), dtype=tf.float32) # zero reward predicted for initial inference

            value_target_distribution = scalar_to_support(outcomes_target[:, 0], support_size=outcome_support_size, min_value=0)
            value_ce = self.cross_entropy(value_target_distribution, value)
            # Scale gradient by the number of unroll steps (See paper appendix Training)
            value_loss = self.scale_gradient(factor=1/trajectory_length)(value_ce)

            policy_ce = self.cross_entropy(policies_target[:, 0], policy_estimate)
            # Scale gradient by the number of unroll steps (See paper appendix Training)
            policy_loss = self.scale_gradient(factor=1/trajectory_length)(policy_ce)

            # ---------------Logging --------------- #
            policy_target = tf.cast(self.clip_probability_dist(policies_target[:, 0]), tf.float32)
            policy_kl_divergence_per_sample = tf.reduce_sum(
                policy_target * tf.math.log(policy_target / self.clip_probability_dist(policy_estimate)), axis=1)
            policy_kls = policy_kls.write(0, tf.reduce_mean(policy_kl_divergence_per_sample, name="kl_mean"))
            policy_ces = policy_ces.write(0, tf.reduce_mean(policy_loss, name="p_loss"))

            expected_value = support_to_scalar_per_player(value, min_value=0, nr_players=4)
            absolute_value_errors = absolute_value_errors.write(0, tf.reduce_mean(tf.abs(expected_value - tf.cast(outcomes_target[:, 0], tf.float32)), name="val_mae"))

            absolute_reward_errors = absolute_reward_errors.write(0, 0)

            entropy = self.calculate_LSE(batch_size, encoded_states)
            latent_space_entropy = latent_space_entropy.write(0, entropy)
            # ---------------Logging --------------- #


            for i in tf.range(trajectory_length - 1):
                next_action = tf.reshape(next_actions[:, i], [-1, 1])
                value, reward, policy_estimate, encoded_states = self.network.recurrent_inference(
                    encoded_states, next_action, training=True)

                # Scale the gradient at the start of the dynamics function (See paper appendix Training)
                encoded_states = self.scale_gradient(factor=1/2)(encoded_states)

                reward_target_distribution = scalar_to_support(rewards_target[:, i+1], support_size=reward_support_size,
                                                               min_value=0)
                reward_ce = self.cross_entropy(reward_target_distribution, reward)
                # Scale gradient by the number of unroll steps (See paper appendix Training)
                reward_loss += self.scale_gradient(factor=1/trajectory_length)(reward_ce)

                value_target_distribution = scalar_to_support(outcomes_target[:, i+1], support_size=outcome_support_size,
                                                              min_value=0)
                value_ce = self.cross_entropy(value_target_distribution, value)
                # Scale gradient by the number of unroll steps (See paper appendix Training)
                value_loss += self.scale_gradient(factor=1/trajectory_length)(value_ce)

                post_terminal_states = tf.cast(tf.reduce_sum(policies_target[:, i + 1], axis=-1) == 0, tf.float32)
                policy_ce = self.cross_entropy(policies_target[:, i + 1], policy_estimate) * (1 - post_terminal_states)
                # Scale gradient by the number of unroll steps (See paper appendix Training)
                policy_loss += self.scale_gradient(factor=1/trajectory_length)(policy_ce)

                # ---------------Logging --------------- #
                policy_target = tf.cast(self.clip_probability_dist(policies_target[:,  i + 1]), tf.float32)
                policy_kl_divergence_per_sample = tf.reduce_sum(
                    policy_target * tf.math.log(policy_target / self.clip_probability_dist(policy_estimate)), axis=1)
                policy_kls = policy_kls.write(i+1, tf.reduce_mean(policy_kl_divergence_per_sample, name="kl_mean"))
                policy_ces = policy_ces.write(i+1, tf.reduce_mean(policy_ce, name="ce_mean"))

                expected_value = support_to_scalar_per_player(value, min_value=0, nr_players=4)
                absolute_value_errors = absolute_value_errors.write(i+1, tf.reduce_mean(tf.abs(expected_value - tf.cast(outcomes_target[:, i+1], tf.float32)), name="val_mae"))
                expected_reward = support_to_scalar_per_player(reward, min_value=0, nr_players=4)
                absolute_reward_errors = absolute_reward_errors.write(i+1, tf.reduce_mean(tf.abs(expected_reward - tf.cast(rewards_target[:, i+1], tf.float32)), name="reard_mae"))

                entropy = self.calculate_LSE(batch_size, encoded_states)
                latent_space_entropy = latent_space_entropy.write(i+1, entropy)
                # ---------------Logging --------------- #

            loss = tf.reduce_mean(
                self.reward_loss_weight * tf.reduce_sum(reward_loss, axis=-1, name="rewards_loss") +
                self.value_loss_weight * tf.reduce_sum(value_loss, axis=-1, name="value_loss") +
                self.policy_loss_weight * policy_loss, name="loss_mean")

        gradients = tape.gradient(loss, self.network.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        gradient_hists = {f"gradients/layer_{i}_{x.name}": g
                          for i, (g, x) in enumerate(zip(gradients, self.network.trainable_variables))}

        # inspired by https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
        squared_weights_sum = tf.reduce_sum([tf.reduce_sum(x ** 2) for x in self.network.trainable_weights])

        mean_features = tf.reduce_mean(tf.reduce_sum(tf.reshape(
            initial_states, (-1,) + self.config.network.feature_extractor.FEATURE_SHAPE), axis=(1, 2)), axis=0)

        return {
            "training/reward_loss": tf.reduce_mean(reward_loss),
            "training/value_loss": tf.reduce_mean(value_loss),
            "training/policy_loss": tf.reduce_mean(policy_loss),
            "training/squared_weights_sum": squared_weights_sum,
            "training/loss": loss,
            **gradient_hists
        }, absolute_reward_errors.stack(), absolute_value_errors.stack(), policy_kls.stack(), policy_ces.stack(),\
               latent_space_entropy.stack(), mean_features

    @staticmethod
    def calculate_LSE(batch_size, encoded_states):
        latent_space = tf.reshape(encoded_states, (batch_size, -1))
        latent_space = (latent_space - tf.reduce_min(latent_space, axis=0,
                                                     keepdims=True)) + 1e-7  # ensure non zero prob for each dimension
        latent_space_dist = latent_space / tf.reduce_sum(latent_space, axis=0, keepdims=True)

        tf.assert_less(tf.reduce_sum(latent_space_dist, axis=0) - 1, 1e-2)

        entropy = - tf.reduce_mean(tf.reduce_sum((latent_space_dist * tf.math.log(latent_space_dist)), axis=0))

        return entropy

    def cross_entropy(self, target, estimate):
        target = tf.cast(target, dtype=tf.float32)
        # clipping of output for ce is important, if not done, will result in exploding gradients
        estimate = self.clip_probability_dist(estimate)
        cross_entropy = -tf.reduce_sum(target * tf.math.log(estimate), axis=-1)

        return cross_entropy

    @staticmethod
    def scale_gradient(factor):
        @tf.custom_gradient
        def f(x):
            def grad(upstream):
                return upstream * tf.cast(factor, tf.float32)
            return  x, grad
        return f

    @staticmethod
    def clip_probability_dist(dist, eps=1e-07):
        return tf.clip_by_value(dist, eps, 1. - eps)
