import gc
import logging
import pickle
import shutil
import traceback
import uuid
from multiprocessing import Queue
from pathlib import Path
from threading import Thread
from time import sleep

import numpy as np

from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.mu_zero.replay_buffer.sum_tree import SumTree


class FileBasedReplayBufferFromFolder:
    def __init__(
            self,
            max_buffer_size: int,
            batch_size: int,
            nr_of_batches: int,
            max_trajectory_length: int,
            min_trajectory_length: int,
            mdp_value: bool,
            td_error: bool,
            valid_policy_target: bool,
            gamma: float,
            game_data_folder: Path,
            episode_data_folder: Path,
            max_samples_per_episode: int,
            min_non_zero_prob_samples: int,
            use_per: bool,
            value_based_per:bool,
            supervised_targets:bool,
            max_updates=20,
            data_file_ending=".jass-data.pkl",
            episode_file_ending=".jass-episode.pkl",
            cache_path: Path = None,
            clean_up_files = True,
            clean_up_episodes = False,
            start_sampling = True):
        """
        Expects entries in queue with semantics
        (states, actions, rewards, probs, outcomes)
        """

        self.supervised_targets = supervised_targets
        self.value_based_per = value_based_per
        self.td_error = td_error
        self.use_per = use_per
        self.valid_policy_target = valid_policy_target
        self.clean_up_episodes = clean_up_episodes
        self.min_non_zero_prob_samples = min_non_zero_prob_samples
        self.max_samples_per_episode = max_samples_per_episode
        self.episode_data_folder = episode_data_folder
        self.episode_file_ending = episode_file_ending
        self.max_trajectory_length = max_trajectory_length
        self.min_trajectory_length = min_trajectory_length
        self.nr_of_batches = nr_of_batches
        self.gamma = gamma
        self.mdp_value = mdp_value
        self.cache_path = cache_path
        self.clean_up_files = clean_up_files
        self.data_file_ending = data_file_ending
        self.game_data_folder = game_data_folder
        self.max_updates = max_updates
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size

        self.sum_tree = SumTree(capacity=max_buffer_size)

        self.game_data_folder.mkdir(parents=True, exist_ok=True)
        self.episode_data_folder.mkdir(parents=True, exist_ok=True)

        self._size_of_last_update = 0

        self.zero_prob_sample_indices = []

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
        self.update()

        while self.non_zero_samples < self.min_non_zero_prob_samples or self.sum_tree.filled_size < self.batch_size:
            sleep(5)
            logging.info(f"waiting for more samples ({self.non_zero_samples} / {self.min_non_zero_prob_samples}) ..")
            self.update()

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
                        total = self.sum_tree.total()
                        s = np.random.uniform(0, total)
                        idx, priority, identifier = self.sum_tree.get(s, timeout=10)
                        file = self.episode_data_folder / f"{identifier}{self.episode_file_ending}"
                        with open(str(file), "rb") as f:
                            episode = pickle.load(f)

                        trajectory = self._sample_trajectory(episode, sampled_trajectory_length)

                        if self.use_per:
                            P_i = priority / total
                            w_i = (1 / self.batch_size) * (1 / P_i)

                            if not self.value_based_per:
                                priority -= 1
                                self.sum_tree.update(idx, priority)
                        else:
                            w_i = 1

                        if priority == 0:
                            self.zero_prob_sample_indices.append(idx)
                        break
                    except Exception as e:
                        logging.warning(f"CAUGHT ERROR: {e}")

                states.append(trajectory[0]), actions.append(trajectory[1]), rewards.append(trajectory[2])
                probs.append(trajectory[3]), outcomes.append(trajectory[4]), sample_weights.append(w_i)

            sample_weights = np.array(sample_weights)
            if self.use_per:
                sample_weights = sample_weights / np.max(sample_weights)  # only scale updates downwards
            else:
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
        return self.sum_tree.filled_size - len(self.zero_prob_sample_indices)

    @property
    def buffer_size(self):
        return self.sum_tree.filled_size

    def restore(self, tree_from_file: bool):
        if tree_from_file and (self.cache_path / f"replay_buffer.pkl").exists():
            restore_path = self.cache_path / f"replay_buffer.pkl"
            with open(restore_path, "rb") as f:
                self.sum_tree = pickle.load(f)
            logging.info(f"restored replay buffer from {restore_path}")
        else:
            for file in self.episode_data_folder.glob("*"):
                try:
                    with open(file, "rb") as f:
                        states, actions, rewards, probs, values = pickle.load(f)
                except:
                    continue
                priority = self._get_priority(rewards, values)
                self.sum_tree.add(data=file.name.split(".")[0], p=priority)
            logging.info(f"restored replay buffer ({self.sum_tree.filled_size}) from {self.episode_data_folder}")

    def _get_priority(self, rewards, values):
        if self.value_based_per:
            cum_5_step_reward = np.array([rewards[i:i + 5].sum(axis=0) for i in range(values.shape[0])])
            value_n_steps_ahead = np.array([values[i + 5] if i + 5 < values.shape[0] else [0, 0]
                                            for i in range(values.shape[0])])
            priority = np.abs(values - (cum_5_step_reward + value_n_steps_ahead)).mean()
        else:
            priority = self.max_samples_per_episode
        return priority

    def save(self):
        save_path = self.cache_path / f"replay_buffer.pkl"

        if save_path.exists():
            save_path.rename(self.cache_path / f"replay_buffer-old.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(self.sum_tree, f)
            logging.info(f"saved replay buffer to {save_path}")

    def _sample_continuously_from_buffer(self):
        while self.running:
            batches = self._sample_from_buffer(self.nr_of_batches)
            self.sample_queue.put(batches)

            while self.sample_queue.qsize() > 10 and self.running:
                sleep(5)

    def update(self):
        files = list(self.game_data_folder.glob(f"*{self.data_file_ending}"))
        logging.info(f"updating replay buffer, found {len(files)} game data files")
        size_of_last_update = 0
        for file in files:
            try:
                with open(file, "rb") as f:
                    states, actions, rewards, probs, values = pickle.load(f)

                if self.clean_up_files:
                    file.unlink()

                assert len(states) == len(actions) == len(rewards) == len(probs) == len(values)

                for s, a, r, p, v in zip(states, actions, rewards, probs, values):
                    #assert (r.sum(axis=0) == o[0]).all()
                    identifier = str(uuid.uuid4())
                    episode_file = self.episode_data_folder / f"{identifier}{self.episode_file_ending}"
                    with open(str(episode_file), "wb") as f:
                        pickle.dump((s, a, r, p, v), f)

                    if len(self.zero_prob_sample_indices) == 0:
                        old_identifier = self.sum_tree.get_data_at_cursor()
                        old_episode_file = self.episode_data_folder / f"{old_identifier}{self.episode_file_ending}"

                        if old_episode_file.exists() and self.clean_up_files:
                            old_episode_file.unlink()

                        priority = self._get_priority(r, v)
                        self.sum_tree.add(data=identifier, p=priority)
                    else:
                        idx = self.zero_prob_sample_indices.pop(0)
                        old_identifier = self.sum_tree.get_data_at(idx)
                        old_episode_file = self.episode_data_folder / f"{old_identifier}{self.episode_file_ending}"

                        if old_episode_file.exists() and self.clean_up_files:
                            old_episode_file.unlink()

                        self.sum_tree.update(data=identifier,
                                             idx=idx,
                                             p=self.max_samples_per_episode)

                size_of_last_update += len(states)

                del states, actions, rewards, probs, values
                gc.collect()
            except Exception as e:
                logging.warning(f"failed reading file {file}. Exception: {e}, continuing anyway") #, traceback: {traceback.format_exc()}")
                # if self.clean_up_files:
                #    file.unlink()

        logging.info(f"update done, added {size_of_last_update} episodes ")

        self._size_of_last_update += size_of_last_update

    def _sample_trajectory(self, episode, sampled_trajectory_length, i=None):
        states, actions, rewards, probs, values = episode
        episode_length = 37 if states[-1].sum() == 0 else 38

        # assert (rewards.sum(axis=0) == outcomes[0]).all()

        if self.valid_policy_target:
            valid_cards = states.reshape(-1, 36, 45)[:, :, FeaturesSetCppConv.CH_CARDS_VALID]
            valid_trumps = np.ones((states.shape[0], 7))
            valid_trumps[:, -1] *= states.reshape(-1, 36, 45)[:, 0, 44]
            valid_trumps *= states.reshape(-1, 36, 45)[:, 0, 43][:, None]

            valid_actions = np.concatenate((valid_cards, valid_trumps), axis=1)
            valid_actions /= np.maximum(np.nansum(valid_actions, axis=1, keepdims=True), 1)

            probs = valid_actions
            episode = states, actions, rewards, probs, values

        assert np.allclose(probs[:episode_length].sum(axis=-1), 1)

        states, actions, rewards, probs, values = episode
        if self.supervised_targets:
            values = np.array([
                np.sum([
                    x * self.gamma**i for i, x in enumerate(rewards[k:])
                ], axis=0) for k in range(rewards.shape[0])
            ])

            one_hot = np.squeeze(np.eye(probs.shape[-1])[actions.reshape(-1)[:episode_length]])
            probs = one_hot

            episode = states, actions, rewards, probs, values
        elif self.td_error:
            episode = states, actions, rewards, probs, values
        elif self.mdp_value:
            values = np.array([
                np.sum([
                    x * self.gamma**i for i, x in enumerate(rewards[k:])
                ], axis=0) for k in range(rewards.shape[0])
            ])
            episode = states, actions, rewards, probs, values
        else:
            values = np.array([
                np.sum([
                    x * self.gamma**i for i, x in enumerate(rewards)
                ], axis=0) for _ in range(rewards.shape[0])
            ])
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
        if self.clean_up_episodes:
            shutil.rmtree(str(self.episode_data_folder))

