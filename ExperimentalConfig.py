from elections.Ideology import Ideology
import random
from typing import List

import numpy as np

from elections.Candidate import Candidate
from elections.Ideology import Ideology
from elections.NDPopulation import NDPopulation
from elections.PopulationGroup import Independents
from network.Tensor import Tensor


class ExperimentalConfig:
    def __init__(self,
                 ideology_range: float,
                 n_bins: int,
                 model_width: int,
                 model_layers: int,
                 batch_size: int,
                 training_voters: int,
                 sampling_voters: int):

        self.ideology_range = ideology_range
        self.min_ideology = -ideology_range
        self.max_ideology = ideology_range
        self.n_bins = n_bins
        self.model_width = model_width
        self.model_layers = model_layers
        self.batch_size = batch_size
        self.training_voters = training_voters
        self.sampling_voters = sampling_voters

    def gen_random_candidates(self, population: NDPopulation, n: int) -> List[Candidate]:
        candidates = []
        while len(candidates) < n:
            ivec = population.unit_sample_voter().ideology.vec * .5
            if self.min_ideology < ivec[0] < self.max_ideology:
                candidates.append(Candidate(f"c-{len(candidates)}", Independents, Ideology(ivec), 0))

        return candidates

    def convert_bin_to_ideology(self, bin: int) -> float:
        step = self.ideology_range / self.n_bins
        lower = bin * step + self.min_ideology
        upper = lower + step
        return random.uniform(lower, upper)

    def convert_ideology_to_bin(self, ideology: float) -> int:
        ideology = max(self.min_ideology, min(ideology, self.max_ideology))
        pct = (ideology - self.min_ideology) / self.ideology_range * 2
        return int(pct * self.n_bins)

    # note that the return value is the index of the winning candidate and NOT the
    # bin of the winning candidate.
    def convert_candidates_to_input_vec(self, candidates: List[Candidate]) -> Tensor:
        cc = [self.convert_ideology_to_bin(c.ideology.vec[0]) for c in candidates]
        x = np.zeros(shape=(1, self.n_bins), dtype=np.single)
        for c in cc:
            x[0, c] = 1
        return x

    def create_training_sample(self, candidates: List[Candidate], winner: Candidate) -> (int, list[int]):
        w = candidates.index(winner)
        cc = [self.convert_ideology_to_bin(c.ideology.vec[0]) for c in candidates]
        return cc, w
