from pathlib import Path

import numpy as np

from lib.mu_zero.replay_buffer.replay_buffer_from_folder import ReplayBufferFromFolder


def test_buffer_size():
    testee = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        trajectory_length=5,
        data_file_ending=".imperfect.jass-data.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        clean_up_files=False
    )

    assert testee.buffer_size == 28


def test_batch_size():
    testee = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        trajectory_length=5,
        data_file_ending=".imperfect.jass-data.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        clean_up_files=False
    )

    batches = testee.sample_from_buffer(1)
    states, actions, rewards, probs, outcomes = batches[0]

    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == probs.shape[0] == outcomes.shape[0] == 32
    assert states.shape[1] == actions.shape[1] == rewards.shape[1] == probs.shape[1] == outcomes.shape[1]


def test_sample_trajectory():
    testee = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        trajectory_length=5,
        data_file_ending=".imperfect.jass-data.pkl",
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        clean_up_files=False
    )
    testee._update()
    total = testee.sum_tree.total()
    s = np.random.uniform(0, total)
    idx, priority, episode = testee.sum_tree.get(s, timeout=10)

    states, actions, rewards, probs, outcomes = testee._sample_trajectory(episode, i=37)

    assert probs[-1, :].sum() == 0 and rewards[-1, :].sum() == 0 and outcomes[-1, :].sum() == 157