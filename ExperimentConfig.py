import random
from typing import List

import numpy as np

from elections.Candidate import Candidate
from elections.Ideology import Ideology
from elections.NDPopulation import NDPopulation
from elections.PopulationGroup import Independents
from network.Tensor import Tensor
from elections.ElectionConstructor import ElectionConstructor, construct_irv, construct_h2h, construct_plurality


class ExperimentConfig:
    def __init__(self,
                 election_name: str,
                 training_cycles: float = 1000,
                 ideology_range: float = 1.5,
                 ideology_flexibility: float = .7,
                 n_bins: int = 21,
                 model_width: int = 512,
                 model_layers: int = 3,
                 memory_size: int = 20000,
                 batch_size: int = 2048,
                 training_voters: int = 1000,
                 sampling_voters: int = 1000,
                 quality_variance: float = 0,
                 path: str = "none"):

        if election_name == "IRV":
            self.election_constructor = ElectionConstructor(construct_irv, "IRV")
        elif election_name == "H2H":
            self.election_constructor = ElectionConstructor(construct_h2h, "H2H")
        elif election_name == "Plurality":
            self.election_constructor = ElectionConstructor(construct_plurality, "Plurality")
        else:
            assert (False, f"Unrecognized election type {election_name}")

        self.election_name = election_name
        self.training_cycles = training_cycles
        self.ideology_dim = 1
        self.ideology_range = ideology_range
        self.ideology_flexibility = ideology_flexibility
        self.min_ideology = -ideology_range
        self.max_ideology = ideology_range
        self.n_bins = n_bins
        self.model_width = model_width
        self.model_layers = model_layers
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.training_voters = training_voters
        self.sampling_voters = sampling_voters
        self.population = self.create_population()
        self.quality_variance = quality_variance
        self.path = path

    def save(self):
        with open(f"{self.path}.txt", "w") as f:
            f.write(self.toString())

    def toString(self) -> str:
        return f"election_name {self.election_name}\n" + \
               f"training_cycles {self.training_cycles}\n" + \
               f"ideology_range {self.ideology_range}\n" + \
               f"ideology_flexibility {self.ideology_flexibility}\n" + \
               f"n_bins {self.n_bins}\n" + \
               f"model_width {self.model_width}\n" + \
               f"model_layers {self.model_layers}\n" + \
               f"memory_size {self.memory_size}\n" + \
               f"batch_size {self.batch_size}\n" + \
               f"training_voters {self.training_voters}\n" + \
               f"sampling_voters {self.sampling_voters}\n" + \
               f"path {self.path}\n"

    def convert_candidates_to_input_vec(self, candidates: List[Candidate]) -> Tensor:
        cc = [self.convert_ideology_to_bin(c.ideology.vec[0]) for c in candidates]
        x = np.zeros(shape = (1, self.n_bins), dtype = np.single)
        for c in cc:
            x[0, c] = 1
        return x

    def gen_candidates(self, n: int):
        cc = self.gen_random_candidates(n)
        ideologies = [c.ideology.vec[0] for c in cc]
        min_ideology = min(ideologies)
        max_ideology = max(ideologies)

        span = .25
        if min_ideology > -span or max_ideology < span:
            return self.gen_candidates(n)
        else:
            return cc

    def gen_random_candidates(self, n: int) -> List[Candidate]:
        candidates = []
        while len(candidates) < n:
            ivec = self.population.unit_sample_voter().ideology.vec * .5
            if self.min_ideology < ivec[0] < self.max_ideology:
                quality = random.normalvariate(0, self.quality_variance)
                candidates.append(Candidate(f"c-{len(candidates)}", Independents, Ideology(ivec), quality))

        return candidates

    def convert_bin_to_ideology(self, bin: int) -> float:
        step = (self.ideology_range * 2) / self.n_bins
        lower = bin * step + self.min_ideology
        upper = lower + step
        return random.uniform(lower, upper)

    def convert_ideology_to_bin(self, ideology: float) -> int:
        ideology = max(self.min_ideology, min(ideology, self.max_ideology))
        pct = (ideology - self.min_ideology) / (self.ideology_range * 2)
        return int(pct * self.n_bins)

    def create_training_sample(self, candidates: List[Candidate], winner: Candidate) -> (int, list[int]):
        w = candidates.index(winner)
        cc = [self.convert_ideology_to_bin(c.ideology.vec[0]) for c in candidates]
        return cc, w

    @staticmethod
    def create_population() -> NDPopulation:
        population_means = np.zeros(shape = (1,))
        population_stddev = np.ones(shape = (1,))
        return NDPopulation(population_means, population_stddev)
