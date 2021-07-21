import time
from contextlib import contextmanager


class Timings(object):
    def __init__(self):
        self.timings = {}
        self.counts = {}

    def add(self, label: str, delta: float):
        self.timings[label] = self.timings.get(label, 0.0) + delta
        self.counts[label] = self.counts.get(label, 0.0) + 1

    def print(self):
        for k in self.timings.keys():
            print("%20s %.0f %8.3f %8.5f" % (k, self.counts[k], self.timings[k], self.timings[k] / self.counts[k]), flush=True)

    def total_time(self):
        return sum(self.timings.values())

    def reset(self):
        self.timings = {}
        self.counts = {}

    @contextmanager
    def time_block(self, name):
        t = time.time()
        yield
        self.add(name, time.time() - t)
