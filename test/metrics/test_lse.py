from pathlib import Path

from jass.features.labels_action_full import LabelSetActionFull

from lib.metrics.lse import LSE
from test.util import get_test_config


def test_lse_0_step():
    config = get_test_config()

    testee = LSE(
        samples_per_calculation=2,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        n_steps_ahead=0
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 1

    del testee

def test_lse_1_step():
    config = get_test_config()

    testee = LSE(
        samples_per_calculation=2,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        n_steps_ahead=1
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 2

    del testee


def test_lse_30_step():
    config = get_test_config()

    testee = LSE(
        samples_per_calculation=2,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path=str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        n_steps_ahead=30
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 31

    del testee


def test_lse_larger_batch():
    config = get_test_config()

    testee = LSE(
        samples_per_calculation=128,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path=str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        n_steps_ahead=16
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 17

    del testee


def test_lse_more_steps_larger_batch_perfect():
    config = get_test_config(cheating=True)

    testee = LSE(
        samples_per_calculation=1024,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "perfect_resnet_random.pd"),
        n_steps_ahead=16
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 17

    del testee