from pathlib import Path

from jass.features.labels_action_full import LabelSetActionFull

from lib.environment.networking.worker_config import WorkerConfig
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.metrics.apao import APAO
from lib.metrics.spkl import SPKL
from test.util import get_test_config


def test_spkl_0_step():
    config = get_test_config()

    testee = SPKL(
        samples_per_calculation=2,
        feature_length=FeaturesSetCppConv.FEATURE_LENGTH,
        feature_shape=FeaturesSetCppConv.FEATURE_SHAPE,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "resnet_random.pd"),
        n_steps_ahead=0
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert result is not None

    del testee

def test_spkl_1_step():
    config = get_test_config()

    testee = SPKL(
        samples_per_calculation=2,
        feature_length=FeaturesSetCppConv.FEATURE_LENGTH,
        feature_shape=FeaturesSetCppConv.FEATURE_SHAPE,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "resnet_random.pd"),
        n_steps_ahead=1
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert result is not None

    del testee


def test_spkl_30_step():
    config = get_test_config()

    testee = SPKL(
        samples_per_calculation=2,
        feature_length=FeaturesSetCppConv.FEATURE_LENGTH,
        feature_shape=FeaturesSetCppConv.FEATURE_SHAPE,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path=str(Path(__file__).parent.parent / "resources" / "resnet_random.pd"),
        n_steps_ahead=30
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert result is not None

    del testee


def test_spkl_larger_batch():
    config = get_test_config()

    testee = SPKL(
        samples_per_calculation=128,
        feature_length=FeaturesSetCppConv.FEATURE_LENGTH,
        feature_shape=FeaturesSetCppConv.FEATURE_SHAPE,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path=str(Path(__file__).parent.parent / "resources" / "resnet_random.pd"),
        n_steps_ahead=16
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert result is not None

    del testee