from CombinedExperiment import CombinedExperiment
from ExperimentConfig import ExperimentConfig
from Experiment import Experiment, RaceResult
from util.snap import snap
from copy import copy
from typing import List
from elections.Candidate import Candidate
from elections.Voter import Voter
from elections.ElectionResult import ElectionResult
from elections.DefaultConfigOptions import unit_election_config
from elections.Ballot import Ballot

from joblib import Parallel, delayed
from CombinedExperiment import ExperimentResult
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "5"

version = "v10"
snap(version)
n_races = 1000
base_config = ExperimentConfig(name="none",
                               election_name="none",
                               training_cycles=200000,
                               ideology_range=1.5,
                               ideology_flexibility=.7,
                               n_bins=21,
                               model_width=512,
                               model_layers=3,
                               memory_size=100000,
                               batch_size=2048,
                               training_voters=400,
                               sampling_voters=10000,
                               quality_variance=0,
                               candidate_variance=0.5,
                               equal_pct_bins=True,
                               model_path="none")

irv_config = copy(base_config)
irv_config.election_name = "IRV"
irv_config.model_path = f"exp/{version}/IRV"
irv_config.name = "IRV"

h2h_config = copy(base_config)
h2h_config.election_name = "H2H"
h2h_config.model_path = f"exp/{version}/H2H"
h2h_config.name = "Condorcet-Minimax"

irv_exp = Experiment(irv_config)
h2h_exp = Experiment(h2h_config)

def run_election(self,
                 candidates: List[Candidate],
                 voters: List[Voter]) -> ElectionResult:
    ballots = [Ballot(v, candidates, unit_election_config) for v in voters]
    process = self.config.election_constructor()
    result = process.run(ballots, set(candidates))
    return result


class RandomCandidateFactory:
    def candidates(self, n):
        return irv_config.gen_random_candidates(n)

class BetterCandidateFactory:
    def candidates(self, n):
        return irv_config.gen_candidates(n)

class ResultCounter:
    def __init__(self):
        self.n_races = 0
        self.n_static_equal = 0
        self.n_static_better = 0
        self.n_strategic_irv_ge = 0

    def add_sample(self, irv_xr: RaceResult, h2h_xr: RaceResult, irv_sr: RaceResult, h2h_sr: RaceResult):
        self.n_races += 1
        if irv_xr.winner_expected_utility == h2h_xr.winner_expected_utility:
            self.n_static_equal += 1
        elif irv_xr.winner_expected_utility > h2h_xr.winner_expected_utility:
            self.n_static_better += 1

        if irv_sr.winner_expected_utility >= h2h_sr.winner_expected_utility:
            self.n_strategic_irv_ge += 1

        if 1 == 2.3:
            print("ixr %6.3f hxr %6.3f isr %6.3f hsr %6.3f" %
                  (
                      1 + irv_xr.winner_expected_utility,
                      1 + h2h_xr.winner_expected_utility,
                      1 + irv_sr.winner_expected_utility,
                      1 + h2h_sr.winner_expected_utility,
                   ),
                  end=''
                  )
            if irv_xr.winner_expected_utility >= h2h_xr.winner_expected_utility:
                print(" xr! ", end = '')

            if irv_sr.winner_expected_utility >= h2h_sr.winner_expected_utility:
                print(" sr! ", end = '')
            print()

    def print(self):
        print("static pct_irv equal %.2f%%  static pct_irv better %.2f%% strategic pct_irv ge %.2f%%" %
              (self.n_static_equal / self.n_races * 100,
               self.n_static_better / self.n_races * 100,
               self.n_strategic_irv_ge / self.n_races * 100))

class CCResult:
    def __init__(self, irv_xr: RaceResult, h2h_xr: RaceResult, irv_sr: RaceResult, h2h_sr: RaceResult):
        self.irv_xr = irv_xr
        self.h2h_xr = h2h_xr

        self.irv_sr = irv_sr
        self.h2h_sr = h2h_sr

def run_one_comparison(candidate_factory, irv_config: ExperimentConfig, h2h_config: ExperimentConfig, n_voters: int) -> CCResult:
    irv_exp = Experiment(irv_config)
    h2h_exp = Experiment(h2h_config)
    irv_exp.config.sampling_voters = n_voters
    h2h_exp.config.sampling_voters = n_voters
    candidates = candidate_factory.candidates(5)
    irv_xr = irv_exp.run_random_race_c(candidates)
    h2h_xr = h2h_exp.run_random_race_c(candidates)

    irv_sr = irv_exp.run_strategic_race_c(candidates)
    h2h_sr = h2h_exp.run_strategic_race_c(candidates)

    return CCResult(irv_xr, h2h_xr, irv_sr, h2h_sr)

def compare_methods(candidate_factory, n_voters: int, n_races: int) -> ResultCounter:
    result_counter = ResultCounter()

    irv_exp.config.sampling_voters = n_voters
    h2h_exp.config.sampling_voters = n_voters

    results: List[CCResult] = Parallel(n_jobs=8)(
        delayed(run_one_comparison)(candidate_factory, irv_config, h2h_config, n_voters) for _ in range(n_races))

    for r in results:
        result_counter.add_sample(r.irv_xr, r.h2h_xr, r.irv_sr, r.h2h_sr)

    return result_counter


for n_voters in [1000]:
    print("randomCandidates with %5d voters " % n_voters, end = '')
    results = compare_methods(RandomCandidateFactory(), n_voters, 1000)
    results.print()
    print("Better Candidates with %5d voters " % n_voters, end = '')
    results = compare_methods(BetterCandidateFactory(), n_voters, 1000)
    results.print()