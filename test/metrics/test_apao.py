from pathlib import Path

from lib.environment.networking.worker_config import WorkerConfig
from lib.metrics.apao import APAO
from test.util import get_test_config


def test_cpp_mcts():
    config = get_test_config()

    config.agent.iterations = 1

    testee = APAO(
        opponent_name="random",
        worker_config=config,
        network_path=str(Path(__file__).parent.parent / "resources" / "resnet_random.pd"),
        parallel_threads=1,
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert 0.4 <= result <= 0.6