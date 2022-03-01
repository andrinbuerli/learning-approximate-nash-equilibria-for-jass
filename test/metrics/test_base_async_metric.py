from pathlib import Path

from lib.environment.networking.worker_config import WorkerConfig
from lib.metrics.base_async_metric import BaseAsyncMetric
from lib.mu_zero.network.network_base import AbstractNetwork


def metric_method(result):
    return result


class MetricMock(BaseAsyncMetric):

    def get_params(self, thread_nr: int, network: AbstractNetwork) -> []:
        return [thread_nr]

    def __init__(self, worker_config: WorkerConfig, network_path: str, parallel_threads: int):
        super().__init__(worker_config, network_path, parallel_threads, metric_method)

    def get_name(self):
        return "mock"


def test_get_result():
    config = WorkerConfig()
    config.network.type = "resnet"

    testee = MetricMock(
        config,
        network_path = str(Path(__file__).parent.parent / "resources" / "resnet_random.pd"),
        parallel_threads=1
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert result == 0


def test_get_result_multi_threaded():
    config = WorkerConfig()
    config.network.type = "resnet"

    testee = MetricMock(
        config,
        network_path = str(Path(__file__).parent.parent / "resources" / "resnet_random.pd"),
        parallel_threads=10
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert result == 4.5