import time
import logging
import numpy


class SumTree:
    """
    Reference implementation:
    https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    Reference paper:
    https://arxiv.org/pdf/1511.05952.pdf
    """

    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.filled_size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s, start_time=None, timeout=10):
        # logging.debug(f"retrieve: {idx}")
        if start_time is None:
            start_time = time.time()

        now = time.time()
        if now - start_time > timeout:
            raise TimeoutError(f"Retrieve took longer than {timeout} seconds.")

        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s, start_time=start_time, timeout=timeout)
        else:
            return self._retrieve(right, s - self.tree[left], start_time=start_time, timeout=timeout)

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        self.filled_size = min(self.capacity, self.filled_size + 1)

    def update(self, idx, p, data=None):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)
        if data is not None:
            dataIdx = idx - self.capacity + 1
            self.data[dataIdx] = data

    def get(self, s, timeout=10):
        # logging.debug("get")
        idx = self._retrieve(0, s, timeout=timeout)
        # logging.debug("finished retrieve")
        dataIdx = idx - self.capacity + 1
        index = int(idx)
        priority = float(self.tree[idx])
        data = self.data[dataIdx]

        return index, priority, data
