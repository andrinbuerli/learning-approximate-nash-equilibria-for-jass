import abc
import logging
import os
import time
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from multiprocessing import Queue, Process
from multiprocessing.pool import ThreadPool

import numpy as np

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_network
from lib.mu_zero.network.network_base import AbstractNetwork


class BaseAsyncMetric:

    def __init__(
            self,
            worker_config: WorkerConfig,
            network_path: str,
            parallel_threads: int,
            metric_method,
            init_method=None):
        self.init_method = init_method
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

        if self.init_method is not None:
            init_vars = self.init_method()

        while True:
            try:
                network.load(self.network_path)

                if self.init_method is None:
                    params = [self.get_params(i, network) for i in range(self.parallel_threads)]
                else:
                    params = [self.get_params(i, network, init_vars) for i in range(self.parallel_threads)]

                results = pool.starmap(self.metric_method, params)

                if len(results) == 1 and type(results[0]) is dict:
                    self.result_queue.put(results[0])
                else:
                    self.result_queue.put(float(np.mean(results)))

            except Exception as e:
                logging.error(f"Encountered error {e}, continuing anyways")
                raise e

    @abc.abstractmethod
    def get_params(self, thread_nr: int, network: AbstractNetwork, init_vars=None) -> []:
        pass

    def get_latest_result(self) -> dict:
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

