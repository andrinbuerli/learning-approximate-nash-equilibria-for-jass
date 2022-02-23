import queue
from multiprocessing import Pipe
from queue import Queue
from threading import Thread

import tensorflow as tf

from lib.mu_zero.network.network_base import AbstractNetwork


class BufferingNetwork(AbstractNetwork):
    def __init__(self, network: AbstractNetwork, buffer_size: int, timeout: float = 1.0):
        super().__init__()
        self.timeout = timeout
        self.buffer_size = buffer_size
        self.executing_network = network
        self._initial_queue = Queue()
        self._recurrent_queue = Queue()

        self._running = True

        self._initial_thread = Thread(target=self._check_initial_queue)
        self._recurrent_thread = Thread(target=self._check_recurrent_queue)

        self._initial_thread.start(), self._recurrent_thread.start()

    def _check_initial_queue(self):
        observations = []
        conns = []

        while self._running:
            timeout_exception = False
            try:
                observation, con = self._initial_queue.get(timeout=self.timeout)
                observations.append(observation)
                conns.append(con)
            except queue.Empty:
                timeout_exception = True

            if len(observations) >= self.buffer_size or (len(observations) > 0 and timeout_exception):
                result = self.executing_network.initial_inference(tf.concat(observations, axis=0))

                for i, conn in enumerate(conns):
                    conn.send([x[i][None] for x in result])
                    conn.close()

                observations.clear(), conns.clear()

    def _check_recurrent_queue(self):
        encoded_states = []
        actions = []
        conns = []

        while self._running:
            timeout_exception = False
            try:
                (o, a), con = self._recurrent_queue.get(timeout=self.timeout)
                encoded_states.append(o)
                actions.append(a)
                conns.append(con)
            except queue.Empty:
                timeout_exception = True

            if len(encoded_states) >= self.buffer_size or (len(encoded_states) > 0 and timeout_exception):
                result = self.executing_network.recurrent_inference(
                    tf.cast(tf.concat(encoded_states, axis=0), tf.float32), tf.concat(actions, axis=0))

                for i, conn in enumerate(conns):
                    conn.send([x[i][None] for x in result])
                    conn.close()

                encoded_states.clear(), conns.clear(), actions.clear()

    def initial_inference(self, observation, return_connection=False):
        assert len(observation.shape) == 4 or len(observation.shape) == 2, "observation must be batched"

        parent_conn, child_conn = Pipe()
        self._initial_queue.put((observation, child_conn))
        return parent_conn.recv() if not return_connection else parent_conn

    def recurrent_inference(self, encoded_state, action, return_connection=False):
        assert len(encoded_state.shape) == 4 or len(encoded_state.shape) == 2, "encoded_state must be batched"
        assert len(action.shape) == 2, "action must be batched"

        parent_conn, child_conn = Pipe()
        self._recurrent_queue.put(((encoded_state, action), child_conn))
        return parent_conn.recv() if not return_connection else parent_conn

    def summary(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def __del__(self):
        self._running = False
        del self._initial_queue
        del self._initial_thread
        del self._recurrent_queue
        del self._recurrent_thread