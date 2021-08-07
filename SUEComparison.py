import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "5"
import random

import numpy as np

from elections.Candidate import Candidate
from ExperimentConfig import ExperimentConfig
from Experiment import Experiment
from CombinedExperiment import CombinedExperiment
from typing import List
from elections.PopulationGroup import Independents
from elections.Ideology import Ideology
from elections.HeadToHeadElection import HeadToHeadElection
from Experiment import RaceResult
from joblib import Parallel, delayed


class SUEComparison:
    def __init__(self, exp: Experiment, candidate_stddev: float, stddev_cap: float, span: bool):
        self.exp = exp
        self.stddev_cap = stddev_cap
        self.candidate_stddev = candidate_stddev
        self.span = span

    def gen_reference_candidates(self, n: int) -> List[Candidate]:
        candidates = []
        while len(candidates) < n:
            ivec = self.exp.config.population.unit_sample_voter().ideology.vec * self.exp.config.candidate_variance
            if abs(ivec[0]) < self.stddev_cap:
                quality = random.normalvariate(0, self.exp.config.quality_variance)
                candidates.append(Candidate(f"c-{len(candidates)}", Independents, Ideology(ivec), quality))
        ii = [c.ideology.vec[0] for c in candidates]
        min_ideology = np.min(ii)
        max_ideology = np.max(ii)
        if self.span and (min_ideology >  -.25 or max_ideology < .25):
            return self.gen_reference_candidates(n)
        return candidates

    def run_reference_election(self) -> RaceResult:
        candidates = self.gen_reference_candidates(5)
        voters = self.exp.config.population.generate_unit_voters(self.exp.config.sampling_voters)
        winner = self.exp.run_election(candidates, voters)
        utilities = self.exp.compute_utilities(winner, candidates, voters)
        return RaceResult(winner, candidates, candidates, utilities, utilities, HeadToHeadElection.count_of_ties > 0)

    def reference_results_par(self, n: int) -> List[RaceResult]:
        winners: List[RaceResult] = Parallel(n_jobs=16)(delayed(self.run_reference_election)() for _ in range(n))
        return winners

    def reference_results(self, n: int) -> List[RaceResult]:
        results = []
        for i in range(n):
            results.append(self.run_reference_election())

        return results

    def compute_reference_SUE(self) -> float:
        results = self.reference_results(100)
        sue = CombinedExperiment.compute_SUE(results)
        print("SUE for reference data is cv: %.2f cap %.2f span %6s % 6.2f" %
              (self.candidate_stddev, self.stddev_cap, self.span, sue))
        return sue


def run_one(cv: float, cap: float, span: bool):
    flexibility = 0
    election_process = "IRV"
    c = ExperimentConfig("v5",
                         election_process,
                         equal_pct_bins=False,
                         candidate_variance=cv,
                         ideology_flexibility = flexibility,
                         sampling_voters=200,
                         model_path=f"exp/v5/{election_process}")

    exp = Experiment(c)
    sue = SUEComparison(exp, cv, cap, span)
    sue.compute_reference_SUE()

def run_scan():
    for  span in [True, False]:
        for cv in [.5, 1.0]:
            for cap in [1.5, 2.0, 2.5, 20]:
                run_one(cv, cap, span)

