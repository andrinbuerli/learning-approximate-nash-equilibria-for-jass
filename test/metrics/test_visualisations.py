from pathlib import Path

from jass.features.labels_action_full import LabelSetActionFull

from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.metrics.sare import SARE
from lib.metrics.visualise_game import GameVisualisation
from test.util import get_test_config


def test_vis():
    config = get_test_config()

    testee = GameVisualisation(
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        mdp_value=False
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 0

    del testee