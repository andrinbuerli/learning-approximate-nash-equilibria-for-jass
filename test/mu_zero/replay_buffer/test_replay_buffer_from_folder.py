from pathlib import Path

from lib.mu_zero.replay_buffer.replay_buffer_from_folder import ReplayBufferFromFolder


def test_buffer_size():
    testee = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        trajectory_length=5,
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        clean_up_files=False
    )

    assert testee.buffer_size == 4


def test_batch_size():
    testee = ReplayBufferFromFolder(
        max_buffer_size=1000,
        batch_size=32,
        trajectory_length=5,
        game_data_folder=Path(__file__).parent.parent.parent / "resources",
        clean_up_files=False
    )

    batches = testee.sample_from_buffer(1)
    states, actions, rewards, probs, outcomes = batches[0]

    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == probs.shape[0] == outcomes.shape[0] == 32
    assert states.shape[1] == actions.shape[1] == rewards.shape[1] == probs.shape[1] == outcomes.shape[1]