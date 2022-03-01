from pathlib import Path

from jass.features.labels_action_full import LabelSetActionFull

from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.metrics.save import SAVE
from test.util import get_test_config


def test_save_0_step():
    config = get_test_config()

    testee = SAVE(
        samples_per_calculation=3,
        feature_length=FeaturesSetCppConv.FEATURE_LENGTH,
        feature_shape=FeaturesSetCppConv.FEATURE_SHAPE,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "resnet_random.pd"),
        n_steps_ahead=0
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 1

    del testee


def test_save_1_step():
    config = get_test_config()

    testee = SAVE(
        samples_per_calculation=3,
        feature_length=FeaturesSetCppConv.FEATURE_LENGTH,
        feature_shape=FeaturesSetCppConv.FEATURE_SHAPE,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "resnet_random.pd"),
        n_steps_ahead=1
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 2

    del testee


def test_save_more_steps_larger_batch():
    config = get_test_config()

    testee = SAVE(
        samples_per_calculation=32,
        feature_length=FeaturesSetCppConv.FEATURE_LENGTH,
        feature_shape=FeaturesSetCppConv.FEATURE_SHAPE,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "resnet_random.pd"),
        n_steps_ahead=16
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 17

    del testee