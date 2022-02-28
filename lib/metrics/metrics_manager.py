from lib.metrics.base_async_metric import BaseAsyncMetric


class MetricsManager:
    def __init__(self, *metrics: [BaseAsyncMetric]):
        self.metrics = metrics

    def get_latest_metric_state(self) -> dict:
        return {
            x.get_name(): x.get_latest_result() for x in self.metrics
        }
