import logging
import traceback
from multiprocessing import Queue
from pathlib import Path
from random import shuffle
from threading import Thread
from time import sleep

import numpy as np
import tensorflow as tf
from jass.features.feature_example_buffer import parse_feature_example
from jass.features.labels_action_full import LabelSetActionFull


class SupervisedReplayBufferFromFolder:
    def __init__(
            self,
            batch_size: int,
            nr_of_batches: int,
            max_trajectory_length: int,
            min_trajectory_length: int,
            mdp_value: bool,
            td_error: bool,
            valid_policy_target: bool,
            gamma: float,
            features,
            max_updates=20,
            cache_path: Path = None,
            start_sampling = True,
            supervised_data_path="/data"):
        """
        Expects entries in queue with semantics
        (states, actions, rewards, probs, outcomes)
        """

        self.features = features
        self.supervised_data_path = supervised_data_path
        self.td_error = td_error
        self.valid_policy_target = valid_policy_target
        self.max_trajectory_length = max_trajectory_length
        self.min_trajectory_length = min_trajectory_length
        self.nr_of_batches = nr_of_batches
        self.gamma = gamma
        self.mdp_value = mdp_value
        self.cache_path = cache_path
        self.max_updates = max_updates
        self.batch_size = batch_size

        self._size_of_last_update = 0

        self.zero_prob_sample_indices = []

        tf_record_files = [str(x.resolve()) for x in Path(supervised_data_path).glob("*.tfrecord")]
        shuffle(tf_record_files)
        ds = tf.data.TFRecordDataset(tf_record_files)
        self.ds = iter(ds.map(lambda x: parse_feature_example(
            x,
            feature_length=38 * features.FEATURE_LENGTH,
            label_length=38 * LabelSetActionFull.LABEL_LENGTH)).repeat().shuffle(100))

        self.sample_queue = Queue()
        self.running = True
        self.sampling_thread = Thread(target=self._sample_continuously_from_buffer)
        if start_sampling:
            self.start_sampling()

    @property
    def size_of_last_update(self):
        size_of_last_update = self._size_of_last_update
        self._size_of_last_update = 0
        return size_of_last_update

    def start_sampling(self):
        self.sampling_thread.start()

    def stop_sampling(self):
        self.running = False
        while self.sampling_thread.is_alive():
            logging.info("Waiting for sampling thread shutdown")

    def sample_from_buffer(self):
        if not self.sampling_thread.is_alive():
            logging.warning("Restarting sampling thread..")
            self.sampling_thread = Thread(target=self._sample_continuously_from_buffer)
            self.start_sampling()

        logging.info(f"waiting for sample from replay buffer... Queue size: {self.sample_queue.qsize()}")
        return self.sample_queue.get()

    def _sample_from_buffer(self, nr_of_batches):
        batches = []
        logging.info("sampling from replay buffer..")
        for _ in range(nr_of_batches):
            states, actions, rewards, probs, outcomes, sample_weights = [], [], [], [], [], []

            if self.max_trajectory_length > self.min_trajectory_length:
                sampled_trajectory_length = np.random.choice(
                    range(self.min_trajectory_length, self.max_trajectory_length))
            else:
                sampled_trajectory_length = self.max_trajectory_length
            for __ in range(self.batch_size):
                while True:
                    try:
                        observations, y = next(self.ds)
                        trajectory = self._sample_trajectory(observations, y, sampled_trajectory_length)
                        w_i = 1
                        break
                    except Exception as e:
                        logging.warning(f"CAUGHT ERROR: {e}, {traceback.print_exc()}")

                states.append(trajectory[0]), actions.append(trajectory[1]), rewards.append(trajectory[2])
                probs.append(trajectory[3]), outcomes.append(trajectory[4]), sample_weights.append(w_i)

            sample_weights = np.array(sample_weights)
            sample_weights = sample_weights / self.batch_size

            batches.append((
                np.stack(states, axis=0),
                np.stack(actions, axis=0),
                np.stack(rewards, axis=0),
                np.stack(probs, axis=0),
                np.stack(outcomes, axis=0),
                sample_weights
            ))

            del states, actions, rewards, probs, outcomes, sample_weights

        logging.info("sampling from replay buffer successful")
        return batches

    @property
    def non_zero_samples(self):
        return -1

    @property
    def buffer_size(self):
        return -1

    def restore(self, tree_from_file: bool):
        pass

    def save(self):
        pass

    def _sample_continuously_from_buffer(self):
        while self.running:
            batches = self._sample_from_buffer(self.nr_of_batches)
            self.sample_queue.put(batches)

            while self.sample_queue.qsize() > 10 and self.running:
                sleep(5)

    def update(self):
        pass

    def _sample_trajectory(self, observations, y, sampled_trajectory_length, i=None):
        observations, y = observations.numpy().reshape(38, -1), y.numpy().reshape(38, -1)

        episode_length = int(observations.max(axis=-1).sum())  # padded states are zeros only

        states = observations[:episode_length]
        actions = y[:, :43].argmax(axis=-1)[:episode_length]

        reshaped = states.reshape((-1,) + self.features.FEATURE_SHAPE)
        current_team = reshaped[:, 0, 0, self.features.CH_PLAYER:self.features.CH_PLAYER + 4].argmax(axis=-1) % 2
        current_teams = 1 - (np.tile([[0, 1]], [episode_length, 1]) == np.repeat(current_team, 2).reshape(-1, 2))
        current_points = (reshaped[:, 0, 0, self.features.CH_POINTS_OWN:(self.features.CH_POINTS_OPP + 1)] * 157).astype(int)
        current_points = np.take_along_axis(current_points, current_teams, axis=1)

        final_points = np.take_along_axis((y[0, 43:45] * 157).astype(int), current_teams[0], axis=0)
        current_points = np.concatenate((current_points, final_points[None]), axis=0)

        rewards = current_points[1:, :] - current_points[:-1, :]

        assert rewards.sum() == 157

        if self.mdp_value:
            values = np.array([
                np.sum([
                    x * self.gamma ** i for i, x in enumerate(rewards[k:])
                ], axis=0) for k in range(rewards.shape[0])
            ])
        else:
            values = np.array([
                np.sum([
                    x * self.gamma ** i for i, x in enumerate(rewards[:])
                ], axis=0) for _ in range(rewards.shape[0])
            ])

        one_hot = np.squeeze(np.eye(43)[actions.reshape(-1)[:episode_length]])
        probs = one_hot
        episode = states, actions, rewards, probs, values

        # create trajectories beyond terminal state
        i = np.random.choice(range(episode_length)) if i is None else i

        indices = [i+j for j in range(sampled_trajectory_length) if i+j <= episode_length-1]
        assert all([x >= 0 for x in indices])
        trajectory = [x[indices] for x in episode]

        if len(indices) < sampled_trajectory_length:
            states, actions, rewards, probs, values = trajectory
            for _ in range(sampled_trajectory_length - len(indices)):
                states = np.concatenate((states, np.zeros_like(states[-1], dtype=np.float32)[np.newaxis]), axis=0)
                actions = np.concatenate((actions, actions[-1][np.newaxis]), axis=0)
                rewards = np.concatenate((rewards, np.zeros_like(rewards[-1], dtype=np.int32)[np.newaxis]), axis=0)
                probs = np.concatenate((probs, np.zeros_like(probs[-1], dtype=np.float32)[np.newaxis]), axis=0)
                if self.mdp_value:
                    values = np.concatenate((values, np.zeros_like(values[-1], dtype=np.int32)[np.newaxis]), axis=0)
                else:
                    values = np.concatenate((values, values[-1][np.newaxis]), axis=0)

            trajectory = states, actions, rewards, probs, values

        return trajectory

    def __del__(self):
        self.stop_sampling()

