import gc
import logging
import pickle
from pathlib import Path

import numpy as np

from lib.mu_zero.replay_buffer.sum_tree import SumTree


class ReplayBufferFromFolder:
    def __init__(
            self,
            max_buffer_size: int,
            batch_size: int,
            trajectory_length: int,
            game_data_folder: Path,
            max_updates=20,
            data_file_ending=".jass-data.pkl",
            cache_path: Path = None,
            clean_up_files = True):
        """
        Expects entries in queue with semantics
        (states, actions, rewards, probs, outcomes)
        """

        self.cache_path = cache_path
        self.trajectory_length = trajectory_length
        self.clean_up_files = clean_up_files
        self.data_file_ending = data_file_ending
        self.game_data_folder = game_data_folder
        self.max_updates = max_updates
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size

        self.sum_tree = SumTree(capacity=max_buffer_size)

        self.size_of_last_update = 0

    def sample_from_buffer(self, nr_of_batches):
        self._update()
        batches = []
        logging.info("sampling from replay buffer..")
        for _ in range(nr_of_batches):
            states, actions, rewards, probs, outcomes = [], [], [], [], []

            for __ in range(self.batch_size):
                while True:
                    try:
                        total = self.sum_tree.total()
                        s = np.random.uniform(0, total)
                        idx, priority, episode = self.sum_tree.get(s, timeout=10)
                        break
                    except TimeoutError as e:
                        logging.warning(f"CAUGHT ERROR: {e}")
                trajectory = self._sample_trajectory(episode)

                states.append(trajectory[0]), actions.append(trajectory[1]), rewards.append(trajectory[2])
                probs.append(trajectory[3]), outcomes.append(trajectory[4])

            batches.append((
                np.stack(states, axis=0),
                np.stack(actions, axis=0),
                np.stack(rewards, axis=0),
                np.stack(probs, axis=0),
                np.stack(outcomes, axis=0),
            ))

            del states, actions, rewards, probs, outcomes

        logging.info("sampling from replay buffer successful")
        return batches

    @property
    def buffer_size(self):
        self._update()
        return self.sum_tree.filled_size

    def restore(self):
        restore_path = self.cache_path / f"replay_buffer.pkl"
        if restore_path.exists():
            with open(restore_path, "rb") as f:
                self.sum_tree = pickle.load(f)
            logging.info(f"restored replay buffer from {restore_path}")

    def save(self):
        save_path = self.cache_path / f"replay_buffer.pkl"

        if save_path.exists():
            save_path.rename(self.cache_path / f"replay_buffer-old.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(self.sum_tree, f)
            logging.info(f"saved replay buffer to {save_path}")

    def _update(self):
        files = list(self.game_data_folder.glob(f"*{self.data_file_ending}"))
        logging.info(f"updating replay buffer, found {len(files)} game data files")
        self.size_of_last_update = 0
        for file in files:
            try:
                with open(file, "rb") as f:
                    states, actions, rewards, probs, outcomes = pickle.load(f)

                if self.clean_up_files:
                    file.unlink()

                for s, a, r, p, o in zip(states, actions, rewards, probs, outcomes):
                    self.sum_tree.add(data=(s, a, r, p, o), p=1)  # no priorities associated with samples yet

                self.size_of_last_update += len(states)

                del states, actions, rewards, probs, outcomes
                gc.collect()
            except:
                logging.warning(f"failed reading file {file}.")

        logging.info(f"update done, added {self.size_of_last_update} episodes ")

    def _sample_trajectory(self, episode, i=None):
        states, actions, rewards, probs, outcomes = episode
        episode_length = 37 if states[-1].sum() == 0 else 38

        assert np.allclose(probs[:episode_length].sum(axis=-1), 1)

        # do not create trajectories beyond terminal state
        i = np.random.choice(range(episode_length)) if i is None else i

        indices = [i+j-1 for j in range(self.trajectory_length) if i+j-1 <= episode_length-1]
        trajectory = [x[indices] for x in episode]

        if len(indices) < self.trajectory_length:
            states, actions, rewards, probs, outcomes = trajectory
            for _ in range(self.trajectory_length - len(indices)):
                states = np.concatenate((states, np.zeros_like(states[-1], dtype=np.float32)[np.newaxis]), axis=0)
                actions = np.concatenate((actions, actions[-1][np.newaxis]), axis=0)
                rewards = np.concatenate((rewards, np.zeros_like(rewards[-1], dtype=np.int32)[np.newaxis]), axis=0)
                probs = np.concatenate((probs, np.zeros_like(probs[-1], dtype=np.float32)[np.newaxis]), axis=0)
                outcomes = np.concatenate((outcomes, outcomes[-1][np.newaxis]), axis=0)

            trajectory = states, actions, rewards, probs, outcomes

        return trajectory


