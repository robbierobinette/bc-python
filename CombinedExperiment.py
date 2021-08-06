from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import os
from Experiment import Experiment, RaceResult
from ExperimentConfig import ExperimentConfig
from elections.Candidate import Candidate
from elections.HeadToHeadElection import HeadToHeadElection


class ExperimentResult:
    def __init__(self,
                 label: str,
                 n_races: int,
                 n_voters: int,
                 sue: float,
                 sue2: float,
                 sue_base: float,
                 s_winners: np.ndarray, s_candidates: np.ndarray,
                 r_winners: np.ndarray, r_candidates: np.ndarray,
                 condorcet_tie_pct: float = 0):
        self.label: str = label
        self.sue = sue
        self.sue2 = sue2
        self.sue_base = sue_base
        self.s_winners = s_winners
        self.s_candidates = s_candidates

        self.r_winners = r_winners
        self.r_candidates = r_candidates
        self.condorcet_tie_pct = condorcet_tie_pct

    @staticmethod
    def compute_stats(ideologies: np.ndarray) -> (float, float):
        sigma = np.mean(np.abs(ideologies))
        representation = [CombinedExperiment.representation(i) for i in ideologies]
        score = np.mean(representation)
        return sigma, score

    def print(self):
        print(self.to_string())

    def to_string(self) -> str:
        r_sigma, r_score = self.compute_stats(self.r_winners)
        s_sigma, s_score = self.compute_stats(self.s_winners)
        return "%14s SUE: % 6.2f SUE2: % 6.2f sue_base: % 6.2f, strategic sigma %5.2f score %5.2f condorcet_ties %4.02f%%" % \
               (self.label, self.sue, self.sue2, self.sue_base, s_sigma, s_score, self.condorcet_tie_pct * 100)

class CombinedExperiment:
    def __init__(self, configs: List[ExperimentConfig], path_base: str, n_races: int):
        self.path_base = path_base
        self.configs = configs
        self.n_races = n_races
        self.names = ["c-0", "c-1", "c-2", "c-3", "c-4"]

        self.experiments = [Experiment(config) for config in configs]

    def run(self) -> List[ExperimentResult]:
        return [self.run_experiment(exp) for exp in self.experiments]

    def estimate_utility(self, sigma: float) -> float:
        import random as r
        count = 1000
        sum = 0.0
        for i in range(count):
            x = r.normalvariate(0, 1)
            sum += abs(x - sigma)
        return - sum/count

    def estimate_SUE(self, stategic_results: List[RaceResult]):
        au = 0
        mu = 0
        wu = 0
        for r in stategic_results:
            bi = [c.ideology.vec[0] for c in r.base_candidates]
            au += np.average([self.estimate_utility(i) for i in bi])
            mu += self.estimate_utility(np.min(np.abs(bi)))
            wu += self.estimate_utility(r.winner.ideology.vec[0])

        return (wu - au) / (mu - au)

    @staticmethod
    def compute_SUE(strategic_results: List[RaceResult]) -> float:
        average_utility = np.average([np.average(r.base_utilities) for r in strategic_results])
        max_utility = np.average([np.max(r.base_utilities) for r in strategic_results])
        winning_utility = np.average([r.utilities[0] for r in strategic_results])
        return (winning_utility - average_utility) / (max_utility - average_utility)

    def run_experiment(self, exp: Experiment) -> ExperimentResult:
        HeadToHeadElection.count_of_ties = 0
        race_results = exp.run_strategic_races(self.n_races)
        # if exp.config.election_name == "H2H":
        #     print(f"number of Condorcet cycles:  {HeadToHeadElection.count_of_ties}")
        random_results = exp.compute_random_results(self.n_races)
        name = exp.config.election_name

        os.system(f"mkdir -p {self.path_base}/plots")

        exp.plot_results(self.winning_ideologies(race_results),
                         f"Frequency of Winning Ideology for Strategic Candidates With {name}",
                         self.names)
        plt.savefig(f"{self.path_base}/plots/strategic_wins_{name}.png")

        exp.plot_results(self.candidate_ideologies(race_results),
                         f"Frequency of Candidate Ideology for Strategic Candidates With {name}", self.names)
        plt.savefig(f"{self.path_base}/plots/strategic_candidates_{name}.png")

        exp.plot_results(self.winning_ideologies(random_results),
                         f"Frequency of Winning Ideology for Random Candidates With {name}",
                         self.names)
        plt.savefig(f"{self.path_base}/plots/random_wins_{name}.png")

        exp.plot_results(self.candidate_ideologies(random_results),
                         f"Frequency of Candidate Ideology for Random Candidates With {name}", self.names)
        plt.savefig(f"{self.path_base}/plots/random_candidates_{name}.png")

        r_sigma, r_score = self.compute_winner_stats(random_results)
        s_sigma, s_score = self.compute_winner_stats(race_results)

        ties = HeadToHeadElection.count_of_ties / self.n_races
        s_winners = np.array([r.winner.ideology.vec[0] for r in race_results])
        r_winners = np.array([r.winner.ideology.vec[0] for r in random_results])

        sc_i = []
        for r in race_results:
            sc_i = sc_i + [c.ideology.vec[0] for c in r.candidates]

        rc_i = []
        for r in race_results:
            rc_i = rc_i + [c.ideology.vec[0] for c in r.candidates]

        sue = self.compute_SUE(race_results)
        sue2 = self.estimate_SUE(race_results)
        sue_base = self.compute_SUE(random_results)
        return ExperimentResult(exp.config.name, len(race_results), exp.config.sampling_voters, sue, sue2, sue_base, s_winners, r_winners, sc_i, rc_i, ties)

    def winning_ideologies(self, results: List[RaceResult]):
        ideologies = []
        for r in results:
            ideologies.append(r.winner.ideology.vec[0])
        return ideologies

    def candidate_ideologies(self, results: List[RaceResult]):
        ideologies = []
        for r in results:
            ci = [c.ideology.vec[0] for c in r.candidates]
            for ii in ci:
                ideologies.append(ii)
        return ideologies

    def results_for_candidate_xxx(self,
                              results: List[Tuple[Candidate, List[Candidate]]],
                              candidate_name: str,
                              wins_only: bool):
        ideologies = []
        for w, cc in results:
            if wins_only and w.name == candidate_name:
                ideologies.append(w.ideology.vec[0])
            elif not wins_only:
                for c in cc:
                    if c.name == candidate_name:
                        ideologies.append(c.ideology.vec[0])

        print(f"found {len(ideologies)} results")
        return ideologies

    @staticmethod
    def cumulative_normal_dist(x_raw: float) -> float:
        if x_raw < 0:
            neg = 1
        else:
            neg = 0

        k = 1.0 / (1.0 + 0.2316419 * abs(x_raw))
        y = ((((1.330274429 * k - 1.821255978) * k + 1.781477937) * k - 0.356563782) * k + 0.319381530) * k
        y = 1.0 - 0.398942280401 * np.exp(-0.5 * (x_raw * x_raw)) * y
        return (1.0 - neg) * y + neg * (1.0 - y)

    @staticmethod
    def representation(sigma: float) -> float:
        pct = CombinedExperiment.cumulative_normal_dist(sigma)
        return 100 * (1 - 2 * abs(.5 - pct))

    @staticmethod
    def compute_winner_stats(results: List[RaceResult]) -> (float, float):
        winners = [r.winner.ideology.vec[0] for r in results]
        sigmas = [abs(x) for x in winners]
        rr = [CombinedExperiment.representation(x) for x in winners]

        mean_sigma = np.mean(sigmas)
        mean_representation = np.mean(rr)
        return mean_sigma, mean_representation
