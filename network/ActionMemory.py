import numpy as np
import tensorflow as tf
from typing import List
import random
from Tensor import Tensor
from ..util.Timings import Timings


class ElectionMemory:
    def __init__(self, depth: int, max_size: int, state_width: int, action_width: int):
        self.max_size = max_size
        self.state: np.array = np.zeros(shape=(0, 1))
        self.action: np.array = np.zeros(shape=(0, 1))
        self.reward: np.array = np.zeros(shape=(0, 1))
        self.depth = depth
        self.idx = 0

    # state is of dim (sample, observation, input_dim)
    def add_sample(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray):
        assert (self.depth == state.shape[1], "depth must match")
        if self.state.shape[0] == 0:
            self.state = np.zeros(shape=(0, state.shape[1], state.shape[2]), dtype=np.single)
            self.action = np.zeros(shape=(0, action.shape[1]), dtype=np.single)
            self.reward = np.zeros(shape=(0, 1), dtype=np.single)

        if self.state.shape[0] < self.max_size:
            self.state = np.concatenate([self.state, state], axis=0)
            self.action = np.concatenate([self.action, action], axis=0)
            self.reward = np.concatenate([self.reward, reward], axis=0)
        else:
            i = self.idx
            self.state[i] = state
            self.action[i] = action
            self.reward[i] = reward
            self.idx = (self.idx + 1) % self.max_size

    def get_batch(self, batch_size) -> (np.ndarray, np.ndarray, np.ndarray):
        indices = np.random.randint(0, self.state.shape[0], batch_size)
        return tf.gather(self.state, indices), tf.gather(self.action, indices), tf.gather(self.reward, indices)