
import tensorflow as tf
import numpy as np

class ResultMemory:
    def __init__(self, max_size: int):
        self.data = np.zeros(shape=(max_size, 5))
        self.max_size = max_size
        self.count = 0
    def add_sample(self, sample: np.ndarray):
        if sample.ndim == 1:
            sample = np.reshape(sample, (1, sample.shape[0]))
        sr = self.count % self.max_size
        self.count += sample.shape[0]

        er = sr + sample.shape[0]
        self.data[sr:er] = sample

    def get_batch(self, batch_size) -> np.ndarray:
        indices = np.random.randint(0, min(self.max_size, self.count), batch_size)
        return tf.gather(self.data, indices)

