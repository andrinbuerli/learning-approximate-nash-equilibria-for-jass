import logging
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.mu_zero.network.network_base import AbstractNetwork


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        action_space_size,
        num_blocks_representation,
        fcn_blocks_representation,
        num_blocks_dynamics,
        fcn_blocks_dynamics,
        num_blocks_prediction,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_hand_layers,
        fc_player_layers,
        support_size,
        players,
        mask_private,
        mask_valid,
        fully_connected,
        network_path=None
    ):
        super().__init__()
        self.fully_connected = fully_connected
        self.mask_valid = mask_valid
        self.mask_private = mask_private
        self.num_blocks_representation = num_blocks_representation
        self.num_blocks_dynamics = num_blocks_dynamics
        self.num_blocks_prediction = num_blocks_prediction
        self.players = players
        self.num_channels = num_channels
        self.observation_shape = observation_shape
        self.action_space_size = action_space_size
        self.support_size = support_size + 1
        block_output_size_reward = reduced_channels_reward * observation_shape[0] * observation_shape[1]

        block_output_size_value = reduced_channels_value * observation_shape[0] * observation_shape[1]

        block_output_size_policy = reduced_channels_policy * observation_shape[0] * observation_shape[1]

        if network_path is None:
            self.representation_network = RepresentationNetwork(
                observation_shape=observation_shape,
                num_blocks=num_blocks_representation,
                num_blocks_fully_connected=fcn_blocks_representation,
                num_channels=num_channels,
                fully_connected=fully_connected)
            self.dynamics_network = DynamicsNetwork(
                observation_shape=observation_shape,
                action_space_size=action_space_size,
                players=players,
                num_blocks=num_blocks_dynamics,
                num_blocks_fully_connected=fcn_blocks_dynamics,
                num_channels=num_channels,
                reduced_channels_reward=reduced_channels_reward,
                fc_reward_layers=fc_reward_layers,
                full_support_size=self.support_size,
                block_output_size_reward=block_output_size_reward,
                fully_connected=fully_connected
            )

            self.prediction_network = PredictionNetwork(
                observation_shape=observation_shape,
                players=players,
                action_space_size=action_space_size,
                num_blocks=num_blocks_prediction,
                num_channels=num_channels,
                reduced_channels_value=reduced_channels_value,
                reduced_channels_policy=reduced_channels_policy,
                fc_value_layers=fc_value_layers,
                fc_hand_layers=fc_hand_layers,
                fc_policy_layers=fc_policy_layers,
                fc_player_layers=fc_player_layers,
                full_support_size=self.support_size,
                block_output_size_value=block_output_size_value,
                block_output_size_policy=block_output_size_policy,
                fully_connected=fully_connected
            )
        else:
            self.load(network_path, from_graph=True)

        self._warmup()

    def prediction(self, encoded_state, training=False, inc_player=False):
        policy, value, player, hand = self.prediction_network(encoded_state, training=training)
        if training or inc_player:
            return policy, value, player, hand
        else:
            return policy, value

    def representation(self, observation, training=False):
        encoded_state = self.representation_network(observation, training=training)

        #encoded_state_normalized = self._scale_encoded_state(encoded_state)
        encoded_state_normalized = encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action, training=False):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        if self.fully_connected:
            action_one_hot = tf.reshape(tf.one_hot(action, depth=self.action_space_size), (-1, self.action_space_size))
        else:
            action_one_hot = tf.reshape(
                tf.tile(
                    tf.one_hot(action, depth=self.action_space_size),
                    (1, encoded_state.shape[1] * encoded_state.shape[2], 1)),
                (-1, encoded_state.shape[1], encoded_state.shape[2], self.action_space_size))

        x = tf.concat((encoded_state, action_one_hot), axis=-1)
        next_encoded_state, reward = self.dynamics_network(x, training=training)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        # calculate extremas over the spatial dimensions for each channel
        #next_encoded_state_normalized = self._scale_encoded_state(next_encoded_state)
        next_encoded_state_normalized = next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation, training=False):
        batch_size = tf.shape(observation)[0]
        if self.mask_private:
            index_mask = tf.tile(tf.transpose(tf.reshape(tf.repeat(tf.range(45), 36), (45, 36)))[None], (batch_size, 1, 1))
            mask = tf.math.logical_or(index_mask == FeaturesSetCppConv.CH_CARDS_VALID, index_mask == FeaturesSetCppConv.CH_TRUMP_VALID)
            mask = tf.math.logical_or(mask, index_mask == FeaturesSetCppConv.CH_PUSH_VALID)
            mask = tf.math.logical_or(mask, index_mask == FeaturesSetCppConv.CH_HAND)
            mask = 1.0 - tf.cast(mask, tf.float32)
            observation *= tf.reshape(mask, (batch_size, -1))
        elif self.mask_valid:
            index_mask = tf.tile(tf.transpose(tf.reshape(tf.repeat(tf.range(45), 36), (45, 36)))[None], (batch_size, 1, 1))
            mask = tf.math.logical_or(index_mask == FeaturesSetCppConv.CH_CARDS_VALID, index_mask == FeaturesSetCppConv.CH_TRUMP_VALID)
            mask = tf.math.logical_or(mask, index_mask == FeaturesSetCppConv.CH_PUSH_VALID)
            mask = 1.0 - tf.cast(mask, tf.float32)
            observation *= tf.reshape(mask, (batch_size, -1))

        encoded_state = self.representation(observation, training=training)
        policy, value, player, hand = self.prediction(encoded_state, training=training, inc_player=True)
        # reward equal to 0 for consistency
        reward = tf.tile(tf.one_hot(0, depth=self.support_size)[None, None], [batch_size, self.players, 1])
        if training:
            return (
                value,
                reward,
                policy,
                player,
                hand,
                encoded_state,
            )
        else:
            return (
                value,
                reward,
                policy,
                encoded_state,
            )

    def recurrent_inference(self, encoded_state, action, training=False):
        next_encoded_state, reward = self.dynamics(encoded_state, action, training=training)
        policy, value, player, hand = self.prediction(next_encoded_state, training=training, inc_player=True)
        if training:
            return value, reward, policy, player, hand, next_encoded_state
        else:
            return value, reward, policy, next_encoded_state

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.representation_network.save(path / "representation.pd")
        self.dynamics_network.save(path / "dynamics.pd")
        self.prediction_network.save(path / "prediction.pd")

        src = path / f"weights{id(self)}.pkl"
        dest = path / "weights.pkl"
        with open(src, "wb") as f:
            pickle.dump(self.get_weight_list(), f)

        shutil.move(src, dest)

        logging.info(f"saved network at {path}")

    def load(self, path, from_graph=False, save=False):
        path = Path(path)
        assert path.exists()

        if from_graph:
            self.representation_network = tf.keras.models.load_model(path / "representation.pd")
            self.dynamics_network = tf.keras.models.load_model(path / "dynamics.pd")
            self.prediction_network = tf.keras.models.load_model(path / "prediction.pd")
        else:
            src = str(path / "weights.pkl")
            if save:
                dest = str(path / f"weights-{id(self)}.pkl")
                shutil.copy(src, dest)
                with open(dest, "rb") as f:
                    weights = pickle.load(f)
                os.remove(dest)
            else:
                with open(src, "rb") as f:
                    weights = pickle.load(f)

            self.set_weights_from_list(weights)

        logging.info(f"loaded network from {path}")

    def summary(self):
        representation_params = sum([tf.keras.backend.count_params(p) for p in self.representation_network.trainable_weights])
        dynamics_params = sum([tf.keras.backend.count_params(p) for p in self.dynamics_network.trainable_weights])
        prediction_params = sum([tf.keras.backend.count_params(p) for p in self.prediction_network.trainable_weights])
        print("\n", 50*"-", "\nResidual MuZero Network Summary\n", 50*"-",
        f"""
representation_network: {representation_params:,} trainable parameters
dynamics_network: {dynamics_params:,} trainable parameters
prediction_network: {prediction_params:,} trainable parameters

TOTAL: {sum([representation_params, dynamics_params, prediction_params]):,} trainable parameters
        """)

    def _scale_encoded_state(self, encoded_state):
        # Scale encoded state between [0, 1] (See appendix paper Training)
        # calculate extremas over the spatial dimensions for each channel
        minimas = tf.reduce_min(encoded_state, axis=(1, 2), keepdims=True)
        maximas = tf.reduce_max(encoded_state, axis=(1, 2), keepdims=True)
        scale_encoded_state = maximas - minimas
        scale_encoded_state = tf.maximum(scale_encoded_state, 1e-5)
        encoded_state_normalized = (
                                           encoded_state - minimas
                                   ) / scale_encoded_state
        return encoded_state_normalized

    def _warmup(self):
        encoded_state = self.representation(np.random.uniform(0, 1, (1,) + tuple(self.observation_shape)).reshape(1, -1))
        if not self.fully_connected:
            assert encoded_state.shape == (1, self.observation_shape[0], self.observation_shape[1], self.num_channels)
        else:
            assert encoded_state.shape == (1, self.num_channels)
        encoded_next_state, reward = self.dynamics(encoded_state, action=np.array([[1]]))
        if not self.fully_connected:
            assert encoded_next_state.shape == (1, self.observation_shape[0], self.observation_shape[1], self.num_channels)
        else:
            assert encoded_next_state.shape == (1, self.num_channels)
        assert reward.shape == (1, self.players, self.support_size)
        policy, value = self.prediction(encoded_next_state)
        assert policy.shape == (1, self.action_space_size)
        assert value.shape == (1, self.players, self.support_size)

    def __del__(self):
        del self


class RepresentationNetwork(tf.keras.Model):
    def __init__(
        self,
        observation_shape,
        num_blocks,
        num_blocks_fully_connected,
        num_channels,
        fully_connected
    ):
        super().__init__()

        self.fully_connected = fully_connected
        self.observation_shape = observation_shape
        self.layer0 = conv2x3(num_channels) if not fully_connected else dense(num_channels)
        self.bn = layers.BatchNormalization()
        self.resblocks = [ResidualBlock(num_channels, fully_connected) for _ in range(num_blocks)]
        self.resblocks_fcn = [ResidualFullyConnectedBlock(num_channels, fully_connected) for _ in range(num_blocks_fully_connected)]

    def call(self, x, training=None):

        if not self.fully_connected:
            x = tf.reshape(x, (-1, self.observation_shape[0], self.observation_shape[1], self.observation_shape[2]))

        x = self.layer0(x, training=training)
        x = self.bn(x, training=training)
        x = tf.nn.tanh(x)

        for block in self.resblocks:
            x = block(x, training=training)

        for block in self.resblocks_fcn:
            x = block(x, training=training)

        return x


class DynamicsNetwork(tf.keras.Model):
    def __init__(
        self,
        observation_shape,
        action_space_size,
        players,
        num_blocks,
        num_blocks_fully_connected,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
        fully_connected
    ):
        super().__init__()
        self.fully_connected = fully_connected
        self.action_space_size = action_space_size
        self.players = players
        self.full_support_size = full_support_size
        self.observation_shape = observation_shape
        self.num_channels = num_channels
        self.layer0 = conv2x3(num_channels) if not fully_connected else dense(num_channels)
        self.bn = layers.BatchNormalization()
        self.resblocks = [ResidualBlock(num_channels, fully_connected) for _ in range(num_blocks)]
        self.resblocks_fcn = [ResidualFullyConnectedBlock(num_channels, fully_connected) for _ in range(num_blocks_fully_connected)]

        self.conv1x1_reward =  layers.Conv2D(filters=reduced_channels_reward, kernel_size=(1, 1), padding="same",
                                             activation=None, use_bias=False, kernel_initializer="glorot_uniform") if not fully_connected else dense(fc_reward_layers[0])
        self.block_output_size_reward = block_output_size_reward if not fully_connected else fc_reward_layers[0]
        self.fc_reward = [
                mlp(
                self.block_output_size_reward, fc_reward_layers, full_support_size,
                output_activation=layers.Activation("softmax"),
                name=f"reward_{_}"
            ) for _ in range(players // 2)
        ]

    def call(self, x, training=None):

        if not self.fully_connected:
            x = tf.reshape(x, (-1, self.observation_shape[0], self.observation_shape[1], self.num_channels + self.action_space_size))

        x = self.layer0(x, training=training)
        x = self.bn(x, training=training)
        x = tf.nn.tanh(x)

        for block in self.resblocks:
            x = block(x, training=training)

        for block in self.resblocks_fcn:
            x = block(x, training=training)

        state = x
        x = tf.nn.tanh(self.conv1x1_reward(state, training=training))
        x = tf.reshape(x, (-1, self.block_output_size_reward))
        reward = tf.tile(tf.stack(([fc(x) for fc in self.fc_reward]), axis=1), [1, 2, 1])
        return state, reward


class PredictionNetwork(tf.keras.Model):
    def __init__(
        self,
        observation_shape,
        players,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_hand_layers,
        fc_player_layers,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
        fully_connected
    ):
        super().__init__()
        self.fully_connected = fully_connected
        self.players = players
        self.full_support_size = full_support_size
        self.observation_shape = observation_shape
        self.num_channels = num_channels
        self.resblocks = [ResidualBlock(num_channels, fully_connected) for _ in range(num_blocks)]


        self.conv1x1_value = layers.Conv2D(filters=reduced_channels_value, kernel_size=(1, 1), padding="same",
                                           activation=None, use_bias=False, kernel_initializer="glorot_uniform")\
                                if not fully_connected else dense(fc_value_layers[0])
        self.conv1x1_policy = layers.Conv2D(filters=reduced_channels_policy, kernel_size=(1, 1), padding="same",
                                           activation=None, use_bias=False, kernel_initializer="glorot_uniform")\
                                if not fully_connected else dense(fc_policy_layers[0])
        self.conv1x1_player = layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same",
                                           activation=None, use_bias=False, kernel_initializer="glorot_uniform")\
                                if not fully_connected else dense(fc_player_layers[0])
        self.conv1x1_hand = layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same",
                                           activation=None, use_bias=False, kernel_initializer="glorot_uniform")\
                                if not fully_connected else dense(fc_hand_layers[0])
        self.block_output_size_value = block_output_size_value if not fully_connected else fc_value_layers[0]
        self.block_output_size_policy = block_output_size_policy if not fully_connected else fc_policy_layers[0]
        self.block_output_size_player = observation_shape[0] * observation_shape[1] * 1 if not fully_connected else fc_player_layers[0]
        self.block_output_size_hand = observation_shape[0] * observation_shape[1] * 1 if not fully_connected else fc_hand_layers[0]
        self.fc_value = [
            mlp(
                self.block_output_size_value, fc_value_layers, full_support_size,
                output_activation=layers.Activation("softmax"),
                name=f"value_{_}"
            ) for _ in range(players // 2)
        ]
        self.fc_policy = mlp(
            self.block_output_size_policy, fc_policy_layers, action_space_size,
            output_activation=layers.Activation('softmax'),
            name="policy"
        )
        self.fc_player = mlp(
            self.block_output_size_player, fc_player_layers, players,
            output_activation=layers.Activation('softmax'),
            name="player"
        )
        self.fc_hand = mlp(
            self.block_output_size_hand, fc_hand_layers, 36,
            output_activation=layers.Activation('sigmoid'),
            name="hand"
        )

    def call(self, x, training=None):

        if not self.fully_connected:
            x = tf.reshape(x, (-1, self.observation_shape[0], self.observation_shape[1], self.num_channels))

        for block in self.resblocks:
            x = block(x, training=training)

        value = tf.nn.tanh(self.conv1x1_value(x, training=training))
        value = tf.reshape(value, (-1, self.block_output_size_value))
        value = tf.tile(tf.stack(([fc(value) for fc in self.fc_value]), axis=1), [1, 2, 1])
        policy = tf.nn.tanh(self.conv1x1_policy(x, training=training))
        policy = tf.reshape(policy, (-1, self.block_output_size_policy))
        policy = self.fc_policy(policy, training=training)
        player = tf.nn.tanh(self.conv1x1_player(x, training=training))
        player = tf.reshape(player, (-1, self.block_output_size_player))
        player = self.fc_player(player, training=training)
        hand = tf.nn.tanh(self.conv1x1_hand(x, training=training))
        hand = tf.reshape(hand, (-1, self.block_output_size_hand))
        hand = self.fc_hand(hand, training=training)
        return policy, value, player, hand


def conv2x3(out_channels, strides=(1, 1), padding='same'):
    return layers.Conv2D(filters=out_channels, kernel_size=(2, 3), strides=strides,
                         padding=padding, activation=None, use_bias=False, kernel_initializer="glorot_uniform")

def conv4x9(out_channels, strides=(1, 1), padding='valid'):
    return layers.Conv2D(filters=out_channels, kernel_size=(4, 9), strides=strides,
                         padding=padding, activation=None, use_bias=False, kernel_initializer="glorot_uniform")

def dense(out_channels):
    return layers.Dense(units=out_channels, activation=None, use_bias=True, kernel_initializer="glorot_uniform")

# Residual block
class ResidualBlock(tf.keras.Model):
    def __init__(self, num_channels, fully_connected):
        super().__init__()
        self.layer1 = conv2x3(num_channels) if not fully_connected else dense(num_channels)
        self.bn1 = layers.BatchNormalization()
        self.layer2 = conv2x3(num_channels) if not fully_connected else dense(num_channels)
        self.bn2 = layers.BatchNormalization()
        self.dense = dense

    def call(self, x, training=None):
        out = self.layer1(x, training=training)
        out = self.bn1(out, training=training)
        out = tf.nn.tanh(out)
        out = self.layer2(out, training=training)
        out = self.bn2(out, training=training)
        out = tf.nn.tanh(out)
        out += x
        return out

class ResidualFullyConnectedBlock(tf.keras.Model):
    def __init__(self, num_channels, fully_connected):
        super().__init__()
        self.layer1 = conv2x3(num_channels // 2) if not fully_connected else dense(num_channels // 2)
        self.bn1 = layers.BatchNormalization()
        self.layer2 = conv4x9(num_channels // 2) if not fully_connected else dense(num_channels // 2)
        self.bn2 = layers.BatchNormalization()
        self.layer3 = layers.Conv2D(filters=num_channels, kernel_size=(1, 1), padding="same",
                                    activation=None, use_bias=False, kernel_initializer="glorot_uniform") \
                    if not fully_connected else dense(num_channels)
        self.bn3 = layers.BatchNormalization()

    def call(self, x, training=None):
        out = self.layer1(x, training=training)
        out = self.bn1(out, training=training)
        out = tf.nn.tanh(out)
        out = self.layer2(out, training=training)
        out = self.bn2(out, training=training)
        out = tf.nn.tanh(out)
        out = self.layer3(out, training=training)
        out = self.bn3(out, training=training)
        out = tf.nn.tanh(out)
        out += x
        return out


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=layers.Activation('softmax'),
    activation=layers.Activation('tanh'),
    name=""
):
    sizes = layer_sizes + [output_size]
    mlp_layers = [layers.Input(shape=(input_size,))]
    for i in range(len(sizes)):
        act = activation if i < len(sizes) - 1 else output_activation
        init = "glorot_uniform" if i < len(sizes) - 1 else "glorot_uniform"
        mlp_layers += [layers.Dense(sizes[i], activation=None, name=f"{name}-dense-{i}", kernel_initializer=init), act]
    return tf.keras.Sequential(mlp_layers)