import gc
import logging
import time
from pathlib import Path

import tensorflow as tf
import wandb
from tqdm import tqdm

from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.log.base_logger import BaseLogger
from lib.metrics.metrics_manager import MetricsManager
from lib.mu_zero.network.network_base import AbstractNetwork
from lib.mu_zero.network.support_conversion import scalar_to_support, support_to_scalar_per_player
from lib.mu_zero.replay_buffer.file_based_replay_buffer_from_folder import FileBasedReplayBufferFromFolder


class MuZeroTrainer:

    def __init__(
            self,
            network: AbstractNetwork,
            replay_buffer: FileBasedReplayBufferFromFolder,
            metrics_manager: MetricsManager,
            config: WorkerConfig,
            logger: BaseLogger,
            optimizer: tf.keras.optimizers.Optimizer,
            min_buffer_size: int,
            updates_per_step: int,
            store_model_weights_after: int,
            player_loss_weight: float = 1.0,
            hand_loss_weight: float = 1.0,
            policy_loss_weight: float = 1.0,
            value_loss_weight: float = 1.0,
            reward_loss_weight: float = 1.0,
            value_entropy_weight: float = 1.0,
            reward_entropy_weight: float = 1.0,
            is_terminal_loss_weight: float = 1.0,
            dldl: bool = False,
            store_weights:bool = True,
            store_buffer:bool = False,
            grad_clip_norm: int = None,
            value_mse: bool = False,
            value_td_5_step: bool = False,
            reward_mse: bool = False,
            log_gradients: bool = True,
            log_inputs: bool = True
    ):
        self.log_inputs = log_inputs
        self.value_td_5_step = value_td_5_step
        self.log_gradients = log_gradients
        self.reward_mse = reward_mse
        self.is_terminal_loss_weight = is_terminal_loss_weight
        self.value_mse = value_mse
        self.dldl = dldl
        self.reward_entropy_weight = reward_entropy_weight
        self.value_entropy_weight = value_entropy_weight
        self.hand_loss_weight = hand_loss_weight
        self.player_loss_weight = player_loss_weight
        self.grad_clip_norm = grad_clip_norm
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

        self.optimizer = optimizer

        channel_mapping = {k: v for k, v in dict(vars(type(self.config.network.feature_extractor))).items() if isinstance(v, int) and v < 100}
        self.feature_names = {}
        for c in range(self.config.network.feature_extractor.FEATURE_SHAPE[-1]):
            for k, v in channel_mapping.items():
                if v > c:
                    break
                self.feature_names[c] = k

    def fit(self, iterations: int, network_path: Path):
        logging.info("Starting alpha zero training")

        while self.replay_buffer.buffer_size < self.min_buffer_size:
            logging.info(f"waiting for buffer to fill up ({self.replay_buffer.buffer_size} / {self.min_buffer_size})")
            self.replay_buffer.update()
            time.sleep(5)

            if self.store_buffer:
                self.replay_buffer.save()

        self.replay_buffer.start_sampling()

        size_of_last_update_cumsum = 0
        for it in tqdm(range(iterations)):
            start = time.time()
            buffer_size = self.replay_buffer.buffer_size
            non_zero_samples = self.replay_buffer.non_zero_samples
            size_of_last_update = self.replay_buffer.size_of_last_update
            size_of_last_update_cumsum += size_of_last_update
            logging.info(f"Iteration {it}, Buffer size: {buffer_size}, Non zero  samples: {non_zero_samples}")

            batches = self.replay_buffer.sample_from_buffer()

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
                if self.log_gradients:
                    grad_infos = {
                        key.replace('/', '_').replace("gradients_", "gradients/"):
                            wandb.Histogram(training_info[key].numpy())
                        for key in training_info if key.__contains__("gradients/")}
                else:
                    grad_infos = {}

                train_infos = {key: float(training_info[key].numpy())
                               for key in training_info if not key.__contains__("gradients/")}

                data = {
                    "meta/buffer_size": buffer_size,
                    "meta/non_zero_samples": non_zero_samples,
                    "meta/sum_tree_total": self.replay_buffer.sum_tree.total(),
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
        for states, actions, rewards, probs, outcomes, priorities in batches:
            info, absolute_reward_errors, absolute_value_errors, policy_kls, policy_ces, ls_entropies, player_ces, \
            hand_ces, policy_entropy, estimated_policy_entropy, value_entropies, reward_entropies, is_terminal_ces, ft = self.train_step(
                tf.convert_to_tensor(states.astype("float32")),
                tf.convert_to_tensor(actions.astype("int32")),
                tf.convert_to_tensor(rewards.astype("int32")),
                tf.convert_to_tensor(probs.astype("float32")),
                tf.convert_to_tensor(outcomes.astype("int32")),
                tf.convert_to_tensor(priorities.astype("float32")))

            reward_error = {
                f"ARE/absolute_reward_error_{i}_steps_ahead": x for i, x in enumerate(absolute_reward_errors)
            }

            value_error = {
                f"AVE/absolute_value_error_{i}_steps_ahead": x for i, x in enumerate(absolute_value_errors)
            }

            value_entropies = {
                f"VEntropy/value_entropy_{i}_steps_ahead": x for i, x in enumerate(value_entropies)
            }

            reward_entropies = {
                f"REntropy/reward_entropy_{i}_steps_ahead": x for i, x in enumerate(reward_entropies)
            }

            policy_kls = {
                f"PKL/policy_kl_{i}_steps_ahead": x for i, x in enumerate(policy_kls)
            }

            policy_ces = {
                f"PCE/policy_ce_{i}_steps_ahead": x for i, x in enumerate(policy_ces)
            }

            is_terminal_ces = {
                f"ITBCE/is_terminal_bce_{i}_steps_ahead": x for i, x in enumerate(is_terminal_ces)
            }

            policy_entropy = {
                f"TPE/policy_entropy_{i}_steps_ahead": x for i, x in enumerate(policy_entropy)
            }

            estimated_policy_entropy = {
                f"EPE/estimated_policy_entropy_{i}_steps_ahead": x for i, x in enumerate(estimated_policy_entropy)
            }

            player_ces = {
                f"PlayerCE/player_ce_{i}_steps_ahead": x for i, x in enumerate(player_ces)
            }

            hand_ces = {
                f"HBCE/hand_bce_{i}_steps_ahead": x for i, x in enumerate(hand_ces)
            }

            ls_entropies = {
                f"LSE/latent_space_entropy_{i}_steps_ahead": x for i, x in enumerate(ls_entropies)
            }

            train_input_dict = {
                f"train_input/channel_{c}_{self.feature_names[c]}": ft[c] for c in range(self.config.network.feature_extractor.FEATURE_SHAPE[-1])
            } if self.log_inputs else {}

            training_infos.append({
                **info, **reward_error, **value_error, **policy_kls, **policy_ces, **ls_entropies,
                **train_input_dict, **player_ces, **hand_ces, **policy_entropy, **estimated_policy_entropy,
                **value_entropies, **reward_entropies, **is_terminal_ces
            })

            del info, absolute_reward_errors, absolute_value_errors, policy_kls, policy_ces

        return training_infos


    def save_latest_network(self, it: int, network_path: Path):
        if self.store_weights:
            logging.info(f'Saving latest model for iteration {it} at {network_path}')
            self.network.save(network_path)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, 2), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, 43), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 2), dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.float32)
        ])
    def train_step(self, states, next_actions, rewards_target, policies_target, outcomes_target, sample_weights):
        batch_size = tf.shape(states)[0]
        trajectory_length = tf.shape(states)[1]

        if self.value_td_5_step:
            trajectory_length = trajectory_length - 5

        policy_kls = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        policy_ces = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        policy_entropy = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        estimated_policy_entropy = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        player_ces = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        hand_bces = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        absolute_value_errors = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        absolute_reward_errors = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        latent_space_entropy = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        reward_entropies = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        value_entropies = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)
        is_terminal_ces = tf.TensorArray(tf.float32, size=trajectory_length, dynamic_size=False, clear_after_read=True)

        rewards_target = tf.tile(rewards_target, [1, 1, 2])
        outcomes_target = tf.tile(outcomes_target, [1, 1, 2])

        with tf.GradientTape() as tape:
            initial_states = states[:, 0]
            post_terminal_states = tf.cast(tf.reduce_sum(initial_states, axis=-1) == 0, tf.float32)
            value, reward, policy_estimate, player, hand, is_terminal, encoded_states = self.network.initial_inference(
                initial_states, training=True)

            reward_support_size = tf.shape(reward)[-1]
            outcome_support_size = tf.shape(value)[-1]

            reshaped_state = tf.reshape(states[:, 0], (-1,) + FeaturesSetCppConv.FEATURE_SHAPE)
            current_player = reshaped_state[:, 0, 0, FeaturesSetCppConv.CH_PLAYER:FeaturesSetCppConv.CH_PLAYER+4]
            player_ce = self.cross_entropy(current_player, player)
            # Scale gradient by the number of unroll steps (See paper appendix Training)
            player_loss = self.scale_gradient(factor=1 / trajectory_length)(player_ce)

            current_hand = tf.reshape(reshaped_state[:, :, :, FeaturesSetCppConv.CH_HAND], (-1, 36))
            hand_bce = self.binary_cross_entropy(current_hand, hand)
            # Scale gradient by the number of unroll steps (See paper appendix Training)
            hand_loss = self.scale_gradient(factor=1 / trajectory_length)(hand_bce)

            is_terminal_bce = self.binary_cross_entropy(post_terminal_states[:, None], is_terminal)
            # Scale gradient by the number of unroll steps (See paper appendix Training)
            is_terminal_loss = self.scale_gradient(factor=1 / trajectory_length)(is_terminal_bce)

            reward_loss = tf.zeros((batch_size, 4), dtype=tf.float32) # zero reward predicted for initial inference

            expected_value = support_to_scalar_per_player(value, min_value=-outcome_support_size//2, nr_players=4)
            if self.value_td_5_step:
                #target_value_5_steps_ahead = self.target_network.initial_inference(states[:, 1 + 5], training=False)[0]
                #target_value_5_steps_ahead = support_to_scalar_per_player(target_value_5_steps_ahead,
                #                                                          min_value=-outcome_support_size//2, nr_players=4)
                # target_value_5_steps_ahead *=  (1 - post_terminal_states[:, None])  # target value after terminal is 0

                target_value_5_steps_ahead = tf.cast(outcomes_target[:, 5], tf.float32)
                cum_reward_5_steps_ahead = tf.cast(tf.reduce_sum(rewards_target[:, :5], axis=1, keepdims=False), tf.float32)
                target_value = cum_reward_5_steps_ahead + target_value_5_steps_ahead
                if self.value_mse:
                    value_ce = (expected_value - target_value) ** 2
                else:
                    value_target_distribution = scalar_to_support(
                        tf.cast(target_value, tf.int32), support_size=outcome_support_size,
                        min_value=-outcome_support_size//2, dldl=self.dldl)
                    value_ce = self.cross_entropy(value_target_distribution, value)
            elif self.value_mse:
                value_ce = (expected_value - tf.cast(outcomes_target[:, 0], tf.float32))**2
            else:
                value_target_distribution = scalar_to_support(
                    outcomes_target[:, 0], support_size=outcome_support_size, min_value=-outcome_support_size//2, dldl=self.dldl)
                value_ce = self.cross_entropy(value_target_distribution, value)
            # Scale gradient by the number of unroll steps (See paper appendix Training)
            value_loss = self.scale_gradient(factor=1/trajectory_length)(value_ce)

            value_H = self.entropy(value)
            value_entropy = value_H

            reward_entropy = tf.zeros((batch_size, 4), dtype=tf.float32) # zero reward predicted for initial inference

            policy_ce = self.cross_entropy(policies_target[:, 0], policy_estimate)
            # Scale gradient by the number of unroll steps (See paper appendix Training)
            policy_loss = self.scale_gradient(factor=1/trajectory_length)(policy_ce)

            # ---------------Logging --------------- #
            policy_target = tf.cast(self.clip_probability_dist(policies_target[:, 0]), tf.float32)
            policy_kl_divergence_per_sample = tf.reduce_sum(
                policy_target * tf.math.log(policy_target / self.clip_probability_dist(policy_estimate)), axis=1)
            policy_kls = policy_kls.write(0, tf.reduce_mean(policy_kl_divergence_per_sample, name="kl_mean"))
            policy_ces = policy_ces.write(0, tf.reduce_mean(player_ce, name="p_loss"))

            policy_entropy = policy_entropy.write(0, tf.reduce_mean(self.entropy(policy_target)))
            estimated_policy_entropy = estimated_policy_entropy.write(0, tf.reduce_mean(self.entropy(policy_estimate)))

            value_entropies = value_entropies.write(0, tf.reduce_mean(value_H))
            reward_entropies = reward_entropies.write(0, 0)

            player_ces = player_ces.write(0, tf.reduce_mean(player_loss, name="player_ces"))

            is_terminal_ces = is_terminal_ces.write(0, tf.reduce_mean(is_terminal_bce, name="is_terminal_bce"))

            hand_bces = hand_bces.write(0, tf.reduce_mean(hand_bce, name="hand_bces"))

            if self.value_td_5_step:
                ave = tf.reduce_mean(tf.abs(expected_value - tf.cast(tf.reduce_sum(rewards_target[:, :], axis=-1), tf.float32)),
                                     name="val_mae")
            else:
                ave = tf.reduce_mean(tf.abs(expected_value - tf.cast(outcomes_target[:, 0], tf.float32)),
                                     name="val_mae")

            absolute_value_errors = absolute_value_errors.write(0, ave)
            absolute_reward_errors = absolute_reward_errors.write(0, 0)

            entropy = self.calculate_LSE(batch_size, encoded_states)
            latent_space_entropy = latent_space_entropy.write(0, entropy)
            # ---------------Logging --------------- #


            for i in tf.range(trajectory_length - 1):
                next_action = tf.reshape(next_actions[:, i], [-1, 1])
                post_terminal_states = tf.cast(tf.reduce_sum(policies_target[:, i+1], axis=-1) == 0, tf.float32)
                value, reward, policy_estimate, player, hand, is_terminal, encoded_states = self.network.recurrent_inference(
                    encoded_states, next_action, training=True)

                # Scale the gradient at the start of the dynamics function (See paper appendix Training)
                encoded_states = self.scale_gradient(factor=1/2)(encoded_states)

                reshaped_state = tf.reshape(states[:, (i + 1)], (-1,) + FeaturesSetCppConv.FEATURE_SHAPE)
                current_player = reshaped_state[:, 0, 0, FeaturesSetCppConv.CH_PLAYER:FeaturesSetCppConv.CH_PLAYER + 4]
                player_ce = self.cross_entropy(current_player, player)
                # Scale gradient by the number of unroll steps (See paper appendix Training)
                player_loss += self.scale_gradient(factor=1 / trajectory_length)(player_ce)

                current_hand = tf.reshape(reshaped_state[:, :, :, FeaturesSetCppConv.CH_HAND], (-1, 36))
                hand_bce = self.binary_cross_entropy(current_hand, hand)
                # Scale gradient by the number of unroll steps (See paper appendix Training)
                hand_loss += self.scale_gradient(factor=1 / trajectory_length)(hand_bce)

                is_terminal_bce = self.binary_cross_entropy(post_terminal_states[:, None], is_terminal)
                # Scale gradient by the number of unroll steps (See paper appendix Training)
                is_terminal_loss += self.scale_gradient(factor=1 / trajectory_length)(is_terminal_bce)

                # predicted reward is associated with action at t-1 therefore index i is used
                # rather than i+1 as for policy and value
                expected_reward = support_to_scalar_per_player(reward, min_value=0, nr_players=4)
                if self.reward_mse:
                    reward_ce = (expected_reward - tf.cast(rewards_target[:, i], tf.float32)) ** 2
                else:
                    reward_target_distribution = scalar_to_support(rewards_target[:, i], support_size=reward_support_size,
                                                                   min_value=0)
                    reward_ce = self.cross_entropy(reward_target_distribution, reward)
                # Scale gradient by the number of unroll steps (See paper appendix Training)
                reward_loss += self.scale_gradient(factor=1/trajectory_length)(reward_ce)

                expected_value = support_to_scalar_per_player(value, min_value=-outcome_support_size//2, nr_players=4)
                if self.value_td_5_step:
                    #target_value_5_steps_ahead = self.target_network.initial_inference(states[:, (i+1)+5], training=False)[0]
                    #target_value_5_steps_ahead = support_to_scalar_per_player(target_value_5_steps_ahead,
                    #                                                          min_value=-outcome_support_size//2, nr_players=4)
                    #target_value_5_steps_ahead *=  (1 - post_terminal_states[:, None])  # target value after terminal is 0

                    target_value_5_steps_ahead = tf.cast(outcomes_target[:, (i+1)+5], tf.float32)

                    cum_reward_5_steps_ahead = tf.cast(tf.reduce_sum(rewards_target[:, (i+1):(i+1)+5], axis=1, keepdims=False), tf.float32)
                    target_value = cum_reward_5_steps_ahead + target_value_5_steps_ahead
                    if self.value_mse:
                        value_ce = (expected_value - target_value) ** 2
                    else:
                        value_target_distribution = scalar_to_support(
                            tf.cast(target_value, tf.int32), support_size=outcome_support_size,
                            min_value=-outcome_support_size//2, dldl=self.dldl)
                        value_ce = self.cross_entropy(value_target_distribution, value)
                elif self.value_mse:
                    value_ce = (expected_value - tf.cast(outcomes_target[:, i+1], tf.float32)) ** 2
                else:
                    value_target_distribution = scalar_to_support(
                        outcomes_target[:, i+1], support_size=outcome_support_size,
                        min_value=-outcome_support_size//2, dldl=self.dldl)
                    value_ce = self.cross_entropy(value_target_distribution, value)
                # Scale gradient by the number of unroll steps (See paper appendix Training)
                value_loss += self.scale_gradient(factor=1/trajectory_length)(value_ce)

                value_H = self.entropy(value)
                value_entropy += value_H

                reward_H = self.entropy(reward)
                reward_entropy += reward_H

                policy_ce = self.cross_entropy(policies_target[:, i+1], policy_estimate) * (1 - post_terminal_states)
                # Scale gradient by the number of unroll steps (See paper appendix Training)
                policy_loss += self.scale_gradient(factor=1/trajectory_length)(policy_ce)

                # ---------------Logging --------------- #
                policy_target = tf.cast(self.clip_probability_dist(policies_target[:,  i+1]), tf.float32)
                policy_kl_divergence_per_sample = tf.reduce_sum(
                    policy_target * tf.math.log(policy_target / self.clip_probability_dist(policy_estimate)), axis=1) * (1 - post_terminal_states)
                policy_kls = policy_kls.write(i+1, tf.reduce_mean(policy_kl_divergence_per_sample, name="kl_mean"))
                policy_ces = policy_ces.write(i+1, tf.reduce_mean(policy_ce, name="ce_mean"))

                policy_entropy = policy_entropy.write(i+1, tf.reduce_mean(self.entropy(policy_target)))
                estimated_policy_entropy = estimated_policy_entropy.write(i+1, tf.reduce_mean(self.entropy(policy_estimate)))

                player_ces = player_ces.write(i+1, tf.reduce_mean(player_ce, name="player_ces"))

                is_terminal_ces = is_terminal_ces.write(i+1, tf.reduce_mean(is_terminal_bce, name="is_terminal_bce"))

                hand_bces = hand_bces.write(i+1, tf.reduce_mean(hand_bce, name="hand_bces"))

                if self.value_td_5_step:
                    ave = tf.reduce_mean(
                        tf.abs(expected_value - tf.cast(tf.reduce_sum(rewards_target[:, i:], axis=-1), tf.float32)),
                        name="val_mae")
                else:
                    ave = tf.reduce_mean(tf.abs(expected_value - tf.cast(outcomes_target[:, i+1], tf.float32)), name="val_mae")

                absolute_value_errors = absolute_value_errors.write(i+1, ave)
                absolute_reward_errors = absolute_reward_errors.write(i+1, tf.reduce_mean(tf.abs(expected_reward - tf.cast(rewards_target[:, i], tf.float32)), name="reard_mae"))

                value_entropies = value_entropies.write(i+1, tf.reduce_mean(value_H))
                reward_entropies = reward_entropies.write(i+1, tf.reduce_mean(reward_H))

                entropy = self.calculate_LSE(batch_size, encoded_states)
                latent_space_entropy = latent_space_entropy.write(i+1, entropy)
                # ---------------Logging --------------- #

            raw_loss = self.reward_loss_weight * tf.reduce_sum(reward_loss, axis=-1, name="rewards_loss")\
                       + self.value_loss_weight * tf.reduce_sum(value_loss, axis=-1,name="value_loss")\
                       + self.policy_loss_weight * policy_loss\
                       + self.player_loss_weight * player_loss\
                       + self.hand_loss_weight * hand_loss\
                       + self.value_entropy_weight * tf.reduce_sum(value_entropy, axis=-1)\
                       + self.reward_entropy_weight * tf.reduce_sum(reward_entropy, axis=-1)\
                       + self.is_terminal_loss_weight * is_terminal_loss
            loss = tf.reduce_sum(
                (
                    raw_loss
                 ) * sample_weights, name="loss_mean")

        gradients = tape.gradient(loss, self.network.trainable_variables)

        if self.grad_clip_norm is not None:
            gradients = [tf.clip_by_norm(grad, clip_norm=self.grad_clip_norm) for grad in gradients]

        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        gradient_hists = {f"gradients/layer_{i}_{x.name}": g
                          for i, (g, x) in enumerate(zip(gradients, self.network.trainable_variables))}

        # inspired by https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
        squared_weights_sum = tf.reduce_sum([tf.reduce_sum(x ** 2) for x in self.network.trainable_weights])

        mean_features = tf.reduce_mean(tf.reduce_sum(tf.reshape(
            initial_states, (-1,) + self.config.network.feature_extractor.FEATURE_SHAPE), axis=(1, 2)), axis=0)

        return {
            "training/reward_loss": tf.reduce_mean(tf.reduce_sum(reward_loss, axis=-1)),
            "training/value_loss": tf.reduce_mean(tf.reduce_sum(value_loss, axis=-1)),
            "training/policy_loss": tf.reduce_mean(policy_loss),
            "training/player_loss": tf.reduce_mean(player_loss),
            "training/reward_entropy": tf.reduce_mean(tf.reduce_sum(reward_entropy, axis=-1)),
            "training/value_entropy": tf.reduce_mean(tf.reduce_sum(value_entropy, axis=-1)),
            "training/hand_loss": tf.reduce_mean(hand_loss),
            "training/is_terminal_loss": tf.reduce_mean(is_terminal_loss),
            "training/squared_weights_sum": squared_weights_sum,
            "training/loss": loss,
            "training/raw_loss": tf.reduce_mean(raw_loss),
            **gradient_hists
        }, absolute_reward_errors.stack(), absolute_value_errors.stack(), policy_kls.stack(), policy_ces.stack(),\
               latent_space_entropy.stack(), player_ces.stack(), hand_bces.stack(), policy_entropy.stack(),\
               estimated_policy_entropy.stack(), value_entropies.stack(), reward_entropies.stack(),\
               is_terminal_ces.stack(), mean_features

    def entropy(self, policy):
        return -tf.reduce_sum(policy * tf.math.log(policy), axis=-1)

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

    def binary_cross_entropy(self, target, estimate):
        target = tf.cast(target, dtype=tf.float32)
        # clipping of output for ce is important, if not done, will result in exploding gradients
        estimate = self.clip_probability_dist(estimate)
        binary_cross_entropy = -tf.reduce_sum(
            target * tf.math.log(estimate) + (1 - target) * tf.math.log(1 - estimate),
            axis=-1)

        return binary_cross_entropy

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
