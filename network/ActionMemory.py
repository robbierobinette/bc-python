from CandidateNetwork import CandidateNetwork
from Candidate import Candidate
import numpy as np
import tensorflow as tf
from typing import List
import random
from Tensor import Tensor
from Timings import Timings


class ActionMemory:
    def __init__(self, max_size: int, state_width: int, action_width: int):
        self.max_size = max_size
        self.state_width = state_width
        self.action_width = action_width
        self.depth_memory = {}
        self.size = 0

    def depths(self) -> List[int]:
        return list(self.depth_memory.keys())

    def ready(self) -> bool:
        return self.size > 1000

    def add_sample(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray):
        self.size += 1
        sample_depth = state.shape[1]
        if sample_depth not in self.depth_memory:
            self.depth_memory[sample_depth] = ActionMemoryDepth(sample_depth, self.max_size, self.state_width,
                                                                self.action_width)

        self.depth_memory[sample_depth].add_sample(state, action, reward)

    def get_batch(self, depth: int, batch_size: int) -> (Tensor, Tensor, Tensor):
        return self.depth_memory[depth].get_batch(batch_size)


class ActionMemoryDepth:
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
