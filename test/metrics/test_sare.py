from pathlib import Path

from jass.features.labels_action_full import LabelSetActionFull

from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.metrics.sare import SARE
from test.util import get_test_config


def test_sare_0_step():
    config = get_test_config()

    testee = SARE(
        samples_per_calculation=3,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        n_steps_ahead=0,
        mdp_value=False
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 0

    del testee


def test_sare_1_step():
    config = get_test_config()

    testee = SARE(
        samples_per_calculation=3,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        n_steps_ahead=1,
        mdp_value=False
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 1

    del testee


def test_sare_more_steps_larger_batch():
    config = get_test_config()

    testee = SARE(
        samples_per_calculation=32,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        n_steps_ahead=16,
        mdp_value=False
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 16

    del testee


def test_sare_more_steps_larger_batch_perfect():
    config = get_test_config(cheating=True)

    testee = SARE(
        samples_per_calculation=32,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "perfect_resnet_random.pd"),
        n_steps_ahead=16,
        mdp_value=False
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 16

    del testee


def test_sare_more_steps_larger_batch_perfect_mdp():
    config = get_test_config(cheating=True)

    testee = SARE(
        samples_per_calculation=32,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "perfect_resnet_random.pd"),
        n_steps_ahead=16,
        mdp_value=True
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 16

    del testee

def test_sare_more_steps_larger_batch_imperfect_mdp():
    config = get_test_config(cheating=False)

    testee = SARE(
        samples_per_calculation=32,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        n_steps_ahead=16,
        mdp_value=True
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 16

    del testee

