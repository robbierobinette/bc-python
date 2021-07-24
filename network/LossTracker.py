import numpy as np


class LossTracker:
    def __init__(self, count):
        self.count = count
        self.losses = np.zeros(shape=(count,))
        self.idx = 0
        self.sum = 0

    def add_loss(self, loss) -> float:
        idx = self.idx % self.count
        if self.idx >= self.count:
            self.sum -= self.losses[idx]
        self.sum += loss
        self.losses[idx] = loss
        self.idx += 1
        return self.sum / min(self.count, self.idx)
