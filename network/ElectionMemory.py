import tensorflow as tf
import numpy as np
from typing import List


class ElectionMemory:
    def __init__(self, max_size: int, n_bins: int):
        self.max_size = max_size
        self.count = 0
        self.x: np.array = np.zeros(shape=(max_size, n_bins), dtype=np.single)
        self.mask: np.array = np.zeros(shape=(max_size, n_bins), dtype=np.single)
        self.y: np.array = np.zeros(shape=(max_size, n_bins), dtype=np.single)
        self.n_bins = n_bins

    # have to deal with the possibility that there will be two candidates
    # with same 'bin' and that one of them will have won.  Thus, we only
    # set the y to 1 for the candidate that won, a nearly identical example
    # will be created for the other candidate where y is 0.
    def add_sample(self, candidates: List[int], winner_index: int):
        for i in range(len(candidates)):
            x = np.zeros(shape=(1, self.n_bins), dtype=np.single)
            y = np.zeros(shape=(1, self.n_bins), dtype=np.single)
            mask = np.zeros(shape=(1, self.n_bins), dtype=np.single)
            for j in range(len(candidates)):
                if i != j:
                    x[0, candidates[j]] = 1

            if i == winner_index:
                y[0, candidates[winner_index]] = 1
            mask[0, candidates[i]] = 1

            self.add_sample_np(x, mask, y)
            self.add_sample_np(np.flip(x, axis=1),
                               np.flip(mask, axis=1),
                               np.flip(y, axis=1)
                               )

    def add_sample_np(self, x: np.ndarray, mask: np.ndarray, y: np.ndarray):
        sr = self.count % self.max_size
        self.count += x.shape[0]

        er = sr + x.shape[0]
        self.x[sr:er] = x
        self.mask[sr:er] = mask
        self.y[sr:er] = y

    def get_batch(self, batch_size) -> (np.ndarray, np.ndarray, np.ndarray):
        indices = np.random.randint(0, min(self.max_size, self.count), batch_size)
        return tf.gather(self.x, indices), tf.gather(self.mask, indices), tf.gather(self.y, indices)
