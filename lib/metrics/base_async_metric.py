import abc
import logging
import os
import time
from multiprocessing import Queue, Process
from multiprocessing.pool import ThreadPool

import numpy as np

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_agent, get_network
from lib.mu_zero.network.network_base import AbstractNetwork


class BaseAsyncMetric:

    def __init__(
            self,
            worker_config: WorkerConfig,
            network_path: str,
            parallel_threads: int,
            metric_method):
        self.metric_method = metric_method
        self.worker_config = worker_config
        self.parallel_threads = parallel_threads
        self.network_path = network_path
        self._latest_result = None
        self.result_queue = Queue()

        self.collecting_process = Process(target=self._calculate_continuously)
        self.collecting_process.start()

    def _calculate_continuously(self):
        while not os.path.exists(self.network_path):
            logging.info(f"waiting for model to be saved at {self.network_path}")
            time.sleep(1)

        pool = ThreadPool(processes=self.parallel_threads)

        network = get_network(self.worker_config)

        while True:
            try:
                network.load(self.network_path)

                params = [self.get_params(i, network) for i in range(self.parallel_threads)]

                results = pool.starmap(self.metric_method, params)

                self.result_queue.put(float(np.mean(results)))
            except Exception as e:
                logging.error(f"Encountered error {e}, continuing anyways")

    @abc.abstractmethod
    def get_params(self, thread_nr: int, network: AbstractNetwork) -> []:
        pass

    def get_latest_result(self):
        while self.result_queue.qsize() > 0:
            self._latest_result = self.result_queue.get()

        return self._latest_result

    def poll_till_next_result_available(self, timeout=.1):
        while self.result_queue.qsize() == 0:
            time.sleep(timeout)

    @abc.abstractmethod
    def get_name(self):
        pass

    def __del__(self):
        self.collecting_process.terminate()
