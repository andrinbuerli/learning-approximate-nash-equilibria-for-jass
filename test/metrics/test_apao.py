from pathlib import Path

from lib.environment.networking.worker_config import WorkerConfig
from lib.metrics.apao import APAO
from test.util import get_test_config


def test_apao_random():
    config = get_test_config()

    config.agent.iterations = 20
    config.agent.n_search_threads = 4

    testee = APAO(
        opponent_name="random",
        worker_config=config,
        network_path=str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        parallel_threads=1,
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()
    print(result)
    assert result is not None

def test_apao_perfect():
    config = get_test_config(cheating=True)

    config.agent.iterations = 20
    config.agent.n_search_threads = 4

    testee = APAO(
        opponent_name="random",
        worker_config=config,
        network_path=str(Path(__file__).parent.parent / "resources" / "perfect_resnet_random.pd"),
        parallel_threads=1,
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()
    print(result)
    assert result is not None