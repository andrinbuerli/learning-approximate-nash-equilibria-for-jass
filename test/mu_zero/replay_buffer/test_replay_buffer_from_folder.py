from pathlib import Path

import numpy as np

from lib.mu_zero.replay_buffer.replay_buffer_from_folder import ReplayBufferFromFolder


def test_buffer_size():
    testee = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        clean_up_files=False,
        mdp_value=False,
        gamma=1
    )

    assert testee.buffer_size > 0

    del testee

def test_batch_size():
    testee = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        clean_up_files=False,
        mdp_value=False,
        gamma=1
    )

    batches = testee.sample_from_buffer()
    states, actions, rewards, probs, outcomes = batches[0]

    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == probs.shape[0] == outcomes.shape[0] == 32
    assert states.shape[1] == actions.shape[1] == rewards.shape[1] == probs.shape[1] == outcomes.shape[1]

    del testee


def test_sample_trajectory():
    testee = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        clean_up_files=False,
        mdp_value=False,
        gamma=1
    )
    testee._update()
    total = testee.sum_tree.total()
    s = np.random.uniform(0, total)
    idx, priority, episode = testee.sum_tree.get(s, timeout=10)

    states, actions, rewards, probs, outcomes = testee._sample_trajectory(episode, i=36, sampled_trajectory_length=5)

    assert probs[-1, :].sum() == 0 and rewards[-1, :].sum() == 0 and outcomes[-1, :].sum() == 157

    del testee


def test_sample_trajectory_mdp_value():
    testee = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        clean_up_files=False,
        mdp_value=True,
        gamma=1
    )
    testee._update()
    total = testee.sum_tree.total()
    s = np.random.uniform(0, total)
    idx, priority, episode = testee.sum_tree.get(s, timeout=10)

    states, actions, rewards, probs, outcomes = testee._sample_trajectory(episode, i=0, sampled_trajectory_length=10)

    assert (outcomes[0] != outcomes[-1]).any()

    del testee

def test_sample_trajectory_mdp_value_game_middle():
    testee = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        clean_up_files=False,
        mdp_value=True,
        gamma=1
    )
    testee._update()
    total = testee.sum_tree.total()
    s = np.random.uniform(0, total)
    idx, priority, episode = testee.sum_tree.get(s, timeout=10)

    states, actions, rewards, probs, outcomes = testee._sample_trajectory(episode, i=16, sampled_trajectory_length=5)

    assert (outcomes[0] != outcomes[-1]).any()

    del testee


def test_sample_trajectory_mdp_value_game_end():
    testee = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        nr_of_batches=1,
        min_trajectory_length=5,
        max_trajectory_length=5,
        data_file_ending=".perfect.jass-data.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        clean_up_files=False,
        mdp_value=True,
        gamma=1
    )
    testee._update()
    total = testee.sum_tree.total()
    s = np.random.uniform(0, total)
    idx, priority, episode = testee.sum_tree.get(s, timeout=10)

    states, actions, rewards, probs, outcomes = testee._sample_trajectory(episode, i=36, sampled_trajectory_length=5)

    assert (outcomes[0] != outcomes[-1]).any()
    assert probs[-1, :].sum() == 0 and rewards[-1, :].sum() == 0 and outcomes[-1, :].sum() == 0

    del testee
