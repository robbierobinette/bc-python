from typing import List
import math
import numpy as np


class IdeologyBase:
    def __init__(self, ideology_vec: np.ndarray):
        self.vec: np.ndarray = ideology_vec


class Ideology(IdeologyBase):
    def __init__(self, ideology_vec: np.ndarray):
        super().__init__(ideology_vec)
        self.dim = ideology_vec.shape[0]


    def distance_from_o(self) -> float:
        dim = self.vec.shape[0]
        return self.distance(Ideology(np.zeros(shape=(dim,))))

    def euclidean_distance(self, rhs: IdeologyBase) -> float:
        deltas = rhs.vec - self.vec
        return np.sqrt(np.sum(deltas * deltas))

    def distance(self, rhs: IdeologyBase) -> float:
        return self.euclidean_distance(rhs)

    # Manhattan distance
    def manhattan_distance(self, rhs: IdeologyBase) -> float:
        l = np.shape(self.vec)[0]
        distance = 0
        for i in range(l):
            distance += abs(self.vec[i] - rhs.vec[i])
        return distance
        # return np.sum(np.abs(self.vec - rhs.vec))
