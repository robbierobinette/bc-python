import random
from typing import List

import numpy as np

from elections.Candidate import Candidate
from elections.Ideology import Ideology
from elections.NDPopulation import NDPopulation
from elections.PopulationGroup import Independents
from network.Tensor import Tensor
from elections.ElectionConstructor import ElectionConstructor, construct_irv, construct_h2h, construct_plurality
from elections.GaussianHelpers import cumulative_normal_dist, rank_to_sigma


class ExperimentConfig:
    def __init__(self,
                 name: str,
                 election_name: str,
                 training_cycles: int = 10000,
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
                 candidate_variance: float = 1.0,
                 equal_pct_bins: bool = False,
                 model_path: str = "none"):

        self.name = name
        self.election_name = election_name
        self.training_cycles = training_cycles
        self.ideology_dim = 1
        self.ideology_range = ideology_range
        self.ideology_flexibility = ideology_flexibility
        self.min_ideology = -ideology_range
        self.max_ideology = ideology_range
        self.n_bins = n_bins
        self.sigma_step = (self.max_ideology - self.min_ideology) / self.n_bins
        self.model_width = model_width
        self.model_layers = model_layers
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.training_voters = training_voters
        self.sampling_voters = sampling_voters
        self.population = self.create_population()
        self.quality_variance = quality_variance
        self.candidate_variance = candidate_variance
        self.equal_pct_bins = equal_pct_bins
        self.model_path = model_path

        self.pct_min = cumulative_normal_dist(self.min_ideology)
        self.pct_max = cumulative_normal_dist(self.max_ideology)
        self.pct_step = (self.pct_max - self.pct_min) / self.n_bins

    def election_constructor(self):
        if self.election_name == "IRV":
            return ElectionConstructor(construct_irv, "IRV")
        elif self.election_name == "H2H":
            return ElectionConstructor(construct_h2h, "H2H")
        elif self.election_name == "Plurality":
            return ElectionConstructor(construct_plurality, "Plurality")
        else:
            assert (False, f"Unrecognized election type {self.election_name}")

    def to_string(self) -> str:
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
               f"path {self.model_path}\n"

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
            ivec = self.population.unit_sample_voter().ideology.vec * self.candidate_variance
            if self.min_ideology < ivec[0] < self.max_ideology:
                quality = random.normalvariate(0, self.quality_variance)
                candidates.append(Candidate(f"c-{len(candidates)}", Independents, Ideology(ivec), quality))

        return candidates


    def convert_bin_to_ideology(self, bin: int) -> float:
        if self.equal_pct_bins:
            return self.convert_bin_to_ideology_pct(bin)
        else:
            return self.convert_bin_to_ideology_sigma(bin)


    def convert_ideology_to_bin(self, bin: int) -> float:
        if self.equal_pct_bins:
            return self.convert_ideology_to_bin_pct(bin)
        else:
            return self.convert_ideology_to_bin_sigma(bin)


    def convert_bin_to_ideology_pct(self, bin: int) -> float:
        pct = self.pct_min + bin * self.pct_step + random.uniform(0, self.pct_step)
        sigma = rank_to_sigma(pct)
        return sigma

    def convert_ideology_to_bin_pct(self, ideology: float) -> int:
        bin = (cumulative_normal_dist(ideology) - self.pct_min) / self.pct_step
        return int(bin)

    def convert_ideology_to_bin_sigma(self, ideology: float) -> int:
        ideology = max(self.min_ideology, min(ideology, self.max_ideology))
        pct = (ideology - self.min_ideology) / (self.ideology_range * 2)
        return int(pct * self.n_bins)

    def convert_bin_to_ideology_sigma(self, bin: int) -> float:
        ideology = self.min_ideology + self.sigma_step * bin + random.uniform(0, self.sigma_step)
        return ideology

    def create_training_sample(self, candidates: List[Candidate], winner: Candidate) -> (int, list[int]):
        w = candidates.index(winner)
        cc = [self.convert_ideology_to_bin(c.ideology.vec[0]) for c in candidates]
        return cc, w

    @staticmethod
    def create_population() -> NDPopulation:
        population_means = np.zeros(shape = (1,))
        population_stddev = np.ones(shape = (1,))
        return NDPopulation(population_means, population_stddev)
