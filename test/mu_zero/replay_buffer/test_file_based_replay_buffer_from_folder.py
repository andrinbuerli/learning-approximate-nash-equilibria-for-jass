import pickle
import uuid
from pathlib import Path
from time import sleep

import numpy as np

from lib.mu_zero.replay_buffer.file_based_replay_buffer_from_folder import FileBasedReplayBufferFromFolder


def test_buffer_size():
    testee = FileBasedReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        episode_file_ending=".perfect.jass-episode.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        episode_data_folder=Path(__file__).parent / f"tmp_episodes{str(uuid.uuid1())}",
        max_samples_per_episode=2,
        min_non_zero_prob_samples=1,
        clean_up_files=False,
        mdp_value=False,
        gamma=1,
        use_per=False,
        valid_policy_target=False,
        clean_up_episodes=True
    )

    sleep(1)

    assert testee.buffer_size > 0

    testee.stop_sampling()
    del testee


def test_batch_size():
    testee = FileBasedReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        episode_file_ending=".perfect.jass-episode.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        episode_data_folder=Path(__file__).parent / f"tmp_episodes{str(uuid.uuid1())}",
        max_samples_per_episode=2,
        min_non_zero_prob_samples=1,
        clean_up_files=False,
        mdp_value=False,
        gamma=1,
        use_per=False,
        valid_policy_target=False,
        clean_up_episodes=True
    )

    sleep(1)

    batches = testee.sample_from_buffer()
    states, actions, rewards, probs, outcomes, sample_weights = batches[0]

    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == probs.shape[0] == outcomes.shape[0] == 32
    assert states.shape[1] == actions.shape[1] == rewards.shape[1] == probs.shape[1] == outcomes.shape[1]

    testee.stop_sampling()
    del testee

def test_min_non_zero_prob_samples():
    testee = FileBasedReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        episode_file_ending=".perfect.jass-episode.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        episode_data_folder=Path(__file__).parent / f"tmp_episodes{str(uuid.uuid1())}",
        max_samples_per_episode=2,
        min_non_zero_prob_samples=1,
        clean_up_files=False,
        mdp_value=False,
        gamma=1,
        use_per=False,
        valid_policy_target=False,
        clean_up_episodes=True
    )

    sleep(1)

    batches = testee.sample_from_buffer()
    states, actions, rewards, probs, outcomes, sample_weights = batches[0]

    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == probs.shape[0] == outcomes.shape[0] == 32
    assert states.shape[1] == actions.shape[1] == rewards.shape[1] == probs.shape[1] == outcomes.shape[1]

    testee.stop_sampling()
    del testee


def test_buffer_restore():
    folder = Path(__file__).parent / f"tmp_episodes{str(uuid.uuid1())}"
    testee1 = FileBasedReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        episode_file_ending=".perfect.jass-episode.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        episode_data_folder=folder,
        max_samples_per_episode=2,
        min_non_zero_prob_samples=1,
        clean_up_files=False,
        mdp_value=False,
        gamma=1,
        use_per=False,
        valid_policy_target=False,
        clean_up_episodes=False
    )

    sleep(1)

    assert testee1.buffer_size > 0

    testee1.stop_sampling()

    testee2 = FileBasedReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        episode_file_ending=".perfect.jass-episode.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        episode_data_folder=folder,
        max_samples_per_episode=2,
        min_non_zero_prob_samples=1,
        clean_up_files=False,
        mdp_value=False,
        gamma=1,
        use_per=False,
        valid_policy_target=False,
        clean_up_episodes=True,
        start_sampling=False
    )

    testee2.restore()

    assert testee2.buffer_size >= testee1.buffer_size

    del testee1, testee2

def test_sample_trajectory():
    testee = FileBasedReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        episode_file_ending=".perfect.jass-episode.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        episode_data_folder=Path(__file__).parent / f"tmp_episodes{str(uuid.uuid1())}",
        max_samples_per_episode=2,
        min_non_zero_prob_samples=1,
        clean_up_files=False,
        mdp_value=False,
        gamma=1,
        use_per=False,
        valid_policy_target=False,
        clean_up_episodes=True
    )

    testee.update()
    total = testee.sum_tree.total()
    s = np.random.uniform(0, total)
    idx, priority, identifier = testee.sum_tree.get(s, timeout=10)
    file = testee.episode_data_folder / f"{identifier}{testee.episode_file_ending}"
    with open(str(file), "rb") as f:
        episode = pickle.load(f)
    states, actions, rewards, probs, outcomes = testee._sample_trajectory(episode, i=36, sampled_trajectory_length=5)

    assert probs[-1, :].sum() == 0 and rewards[-1, :].sum() == 0 and outcomes[-1, :].sum() == 157

    testee.stop_sampling()
    del testee


def test_sample_trajectory_mdp_value():
    testee = FileBasedReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        episode_file_ending=".perfect.jass-episode.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        episode_data_folder=Path(__file__).parent / f"tmp_episodes{str(uuid.uuid1())}",
        max_samples_per_episode=2,
        min_non_zero_prob_samples=1,
        clean_up_files=False,
        mdp_value=True,
        gamma=1,
        use_per=False,
        valid_policy_target=False,
        clean_up_episodes=True
    )

    testee.update()
    total = testee.sum_tree.total()
    s = np.random.uniform(0, total)
    idx, priority, identifier = testee.sum_tree.get(s, timeout=10)
    file = testee.episode_data_folder / f"{identifier}{testee.episode_file_ending}"
    with open(str(file), "rb") as f:
        episode = pickle.load(f)

    states, actions, rewards, probs, outcomes = testee._sample_trajectory(episode, i=0, sampled_trajectory_length=10)

    assert (outcomes[0] != outcomes[-1]).any()

    testee.stop_sampling()
    del testee
