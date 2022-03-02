from lib.metrics.base_async_metric import BaseAsyncMetric


class MetricsManager:
    def __init__(self, *metrics: [BaseAsyncMetric]):
        self.metrics = metrics

    def get_latest_metrics_state(self) -> dict:
        results = [x.get_latest_result() for x in self.metrics]

        return {
            k: v for x in results for (k, v) in x.items()
        }
