import logging
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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
        support_size,
        players,
    ):
        super().__init__()
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

        self.representation_network = RepresentationNetwork(
            observation_shape=observation_shape,
            num_blocks=num_blocks_representation,
            num_blocks_fully_connected=fcn_blocks_representation,
            num_channels=num_channels)

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
                fc_policy_layers=fc_policy_layers,
                full_support_size=self.support_size,
                block_output_size_value=block_output_size_value,
                block_output_size_policy=block_output_size_policy,
            )

        self._warmup()

    def prediction(self, encoded_state, training=False):
        policy, value = self.prediction_network(encoded_state, training=training)
        return policy, value

    def representation(self, observation, training=False):
        encoded_state = self.representation_network(observation, training=training)

        encoded_state_normalized = self._scale_encoded_state(encoded_state)
        return encoded_state_normalized

    def dynamics(self, encoded_state, action, training=False):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = tf.reshape(
            tf.tile(
                tf.one_hot(action, depth=self.action_space_size),
                (1, encoded_state.shape[1] * encoded_state.shape[2], 1)),
            (-1, encoded_state.shape[1], encoded_state.shape[2], self.action_space_size))

        x = tf.concat((encoded_state, action_one_hot), axis=-1)
        next_encoded_state, reward = self.dynamics_network(x, training=training)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        # calculate extremas over the spatial dimensions for each channel
        next_encoded_state_normalized = self._scale_encoded_state(next_encoded_state)

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation, training=False):
        encoded_state = self.representation(observation, training=training)
        policy, value = self.prediction(encoded_state, training=training)
        # reward equal to 0 for consistency
        batch_size = tf.shape(observation)[0]
        reward = tf.tile(tf.one_hot(0, depth=self.support_size)[None, None], [batch_size, self.players, 1])
        return (
            value,
            reward,
            policy,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action, training=False):
        next_encoded_state, reward = self.dynamics(encoded_state, action, training=training)
        policy, value = self.prediction(next_encoded_state, training=training)
        return value, reward, policy, next_encoded_state

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.representation_network.save(path / "representation.pd")
        self.dynamics_network.save(path / "dynamics.pd")
        self.prediction_network.save(path / "prediction.pd")

        with open(path / "weights.pkl", "wb") as f:
            pickle.dump(self.get_weight_list(), f)

        logging.info(f"saved network at {path}")

    def load(self, path):
        path = Path(path)
        assert path.exists()
        self.representation_network = tf.keras.models.load_model(path / "representation.pd")
        self.dynamics_network = tf.keras.models.load_model(path / "dynamics.pd")
        self.prediction_network = tf.keras.models.load_model(path / "prediction.pd")

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
        assert encoded_state.shape == (1, self.observation_shape[0], self.observation_shape[1], self.num_channels)
        encoded_next_state, reward = self.dynamics(encoded_state, action=np.array([[1]]))
        assert encoded_next_state.shape == (1, self.observation_shape[0], self.observation_shape[1], self.num_channels)
        assert reward.shape == (1, self.players, self.support_size)
        policy, value = self.prediction(encoded_next_state)
        assert policy.shape == (1, self.action_space_size)
        assert value.shape == (1, self.players, self.support_size)

    def __del__(self):
        del self.prediction_network, self.representation_network, self.dynamics_network


class RepresentationNetwork(tf.keras.Model):
    def __init__(
        self,
        observation_shape,
        num_blocks,
        num_blocks_fully_connected,
        num_channels
    ):
        super().__init__()

        self.observation_shape = observation_shape
        self.conv = conv2x3(num_channels)
        self.bn = layers.BatchNormalization()
        self.resblocks = [ResidualBlock(num_channels) for _ in range(num_blocks)]
        self.resblocks_fcn = [ResidualFullyConnectedBlock(num_channels) for _ in range(num_blocks_fully_connected)]

    def call(self, x, training=None):
        x = tf.reshape(x, (-1, self.observation_shape[0], self.observation_shape[1], self.observation_shape[2]))

        x = self.conv(x, training=training)
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
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.players = players
        self.full_support_size = full_support_size
        self.observation_shape = observation_shape
        self.num_channels = num_channels
        self.conv = conv2x3(num_channels)
        self.bn = layers.BatchNormalization()
        self.resblocks = [ResidualBlock(num_channels) for _ in range(num_blocks)]
        self.resblocks_fcn = [ResidualFullyConnectedBlock(num_channels) for _ in range(num_blocks_fully_connected)]

        self.conv1x1_reward =  layers.Conv2D(filters=reduced_channels_reward, kernel_size=(1, 1), padding="same",
                                             activation=None, use_bias=False)
        self.block_output_size_reward = block_output_size_reward
        self.fc = mlp(
            self.block_output_size_reward, fc_reward_layers, players * full_support_size,
            output_activation=layers.Activation("linear")
        )

    def call(self, x, training=None):
        x = tf.reshape(x, (-1, self.observation_shape[0], self.observation_shape[1], self.num_channels + self.action_space_size))

        x = self.conv(x, training=training)
        x = self.bn(x, training=training)
        x = tf.nn.tanh(x)

        for block in self.resblocks:
            x = block(x, training=training)

        for block in self.resblocks_fcn:
            x = block(x, training=training)

        state = x
        x = self.conv1x1_reward(x, training=training)
        x = tf.reshape(x, (-1, self.block_output_size_reward))
        reward = self.fc(x, training=training)
        reward = tf.reshape(reward, (-1, self.players, self.full_support_size))
        reward = tf.nn.softmax(reward, axis=-1)
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
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
    ):
        super().__init__()
        self.players = players
        self.full_support_size = full_support_size
        self.observation_shape = observation_shape
        self.num_channels = num_channels
        self.resblocks = [ResidualBlock(num_channels) for _ in range(num_blocks)]


        self.conv1x1_value = layers.Conv2D(filters=reduced_channels_value, kernel_size=(1, 1), padding="same",
                                           activation=None, use_bias=False)
        self.conv1x1_policy = layers.Conv2D(filters=reduced_channels_policy, kernel_size=(1, 1), padding="same",
                                           activation=None, use_bias=False)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size * players,
            output_activation=layers.Activation("linear")
        )
        self.fc_policy = mlp(
            self.block_output_size_policy, fc_policy_layers, action_space_size
        )

    def call(self, x, training=None):
        x = tf.reshape(x, (-1, self.observation_shape[0], self.observation_shape[1], self.num_channels))

        for block in self.resblocks:
            x = block(x, training=training)

        value = self.conv1x1_value(x, training=training)
        policy = self.conv1x1_policy(x, training=training)
        value = tf.reshape(value, (-1, self.block_output_size_value))
        policy = tf.reshape(policy, (-1, self.block_output_size_policy))
        value = self.fc_value(value, training=training)
        value = tf.reshape(value, (-1, self.players, self.full_support_size))
        value = tf.nn.softmax(value, axis=-1)
        policy = self.fc_policy(policy, training=training)
        return policy, value


def conv2x3(out_channels, strides=(1, 1), padding='same'):
    return layers.Conv2D(filters=out_channels, kernel_size=(2, 3), strides=strides,
                         padding=padding, activation=None, use_bias=False)

def conv4x9(out_channels, strides=(1, 1), padding='valid'):
    return layers.Conv2D(filters=out_channels, kernel_size=(4, 9), strides=strides,
                         padding=padding, activation=None, use_bias=False)


# Residual block
class ResidualBlock(tf.keras.Model):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = conv2x3(num_channels)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = conv2x3(num_channels)
        self.bn2 = layers.BatchNormalization()

    def call(self, x, training=None):
        out = self.conv1(x, training=training)
        out = self.bn1(out, training=training)
        out = tf.nn.tanh(out)
        out = self.conv2(out, training=training)
        out = self.bn2(out, training=training)
        out += x
        out = tf.nn.tanh(out)
        return out

class ResidualFullyConnectedBlock(tf.keras.Model):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = conv2x3(num_channels // 2)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = conv4x9(num_channels // 2)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = conv2x3(num_channels)
        self.bn3 = layers.BatchNormalization()

    def call(self, x, training=None):
        out = self.conv1(x, training=training)
        out = self.bn1(out, training=training)
        out = tf.nn.tanh(out)
        out = self.conv2(out, training=training)
        out = self.bn2(out, training=training)
        out = tf.nn.tanh(out)
        out = self.conv3(out, training=training)
        out = self.bn3(out, training=training)
        out += x
        out = tf.nn.tanh(out)
        return out


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=layers.Activation('softmax'),
    activation=layers.Activation('tanh'),
):
    sizes = layer_sizes + [output_size]
    mlp_layers = [layers.Input(shape=(input_size,))]
    for i in range(len(sizes)):
        act = activation if i < len(sizes) - 1 else output_activation
        mlp_layers += [layers.Dense(sizes[i]), act]
    return tf.keras.Sequential(mlp_layers)