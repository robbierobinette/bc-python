from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import os
from Experiment import Experiment
from ExperimentConfig import ExperimentConfig
from elections.Candidate import Candidate
from elections.HeadToHeadElection import HeadToHeadElection


class ExperimentResult:
    def __init__(self,
                 label: str,
                 s_winners: np.ndarray, s_candidates: np.ndarray,
                 r_winners: np.ndarray, r_candidates: np.ndarray,
                 condorcet_tie_pct: float = 0):
        self.label: str = label
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
        return "%14s random sigma %5.2f score %5.2f strategic sigma %5.2f score %5.2f condorcet_ties %4.02f%%" % \
               (self.label, r_sigma, r_score, s_sigma, s_score, self.condorcet_tie_pct)

class CombinedExperiment:
    def __init__(self, configs: List[ExperimentConfig], path_base: str, n_races: int):
        self.path_base = path_base
        self.configs = configs
        self.n_races = n_races
        self.names = ["c-0", "c-1", "c-2", "c-3", "c-4"]

        self.experiments = [Experiment(config) for config in configs]

    def run(self) -> List[ExperimentResult]:
        return [self.run_experiment(exp) for exp in self.experiments]

    def run_experiment(self, exp: Experiment) -> ExperimentResult:
        HeadToHeadElection.count_of_ties = 0
        strategic_results = exp.run_strategic_races_par(self.n_races)
        # if exp.config.election_name == "H2H":
        #     print(f"number of Condorcet cycles:  {HeadToHeadElection.count_of_ties}")
        random_results = exp.compute_random_results(self.n_races)
        name = exp.config.election_name

        os.system(f"mkdir -p {self.path_base}/plots")

        exp.plot_results(self.winning_ideologies(strategic_results),
                         f"Frequency of Winning Ideology for Strategic Candidates With {name}",
                         self.names)
        plt.savefig(f"{self.path_base}/plots/strategic_wins_{name}.png")

        exp.plot_results(self.candidate_ideologies(strategic_results),
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
        s_sigma, s_score = self.compute_winner_stats(strategic_results)

        ties = HeadToHeadElection.count_of_ties / self.n_races
        s_winners = np.array(list(map(lambda x: x[0].ideology.vec[0], strategic_results)))
        r_winners = np.array(list(map(lambda x: x[0].ideology.vec[0], random_results)))

        sc_i = []
        for w, cc in strategic_results:
            sc_i = sc_i + [c.ideology.vec[0] for c in cc]

        rc_i = []
        for w, cc in random_results:
            rc_i = rc_i + [c.ideology.vec[0] for c in cc]

        return ExperimentResult(exp.config.name, s_winners, r_winners, sc_i, rc_i, ties)

    def winning_ideologies(self, results: List[Tuple[Candidate, List[Candidate]]]):
        ideologies = []
        for w, cc in results:
            ideologies.append(w.ideology.vec[0])
        return ideologies

    def candidate_ideologies(self, results: List[Tuple[Candidate, List[Candidate]]]):
        ideologies = []
        for w, cc in results:
            ci = [c.ideology.vec[0] for c in cc]
            for ii in ci:
                ideologies.append(ii)
        return ideologies

    def results_for_candidate(self,
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
    def cumulative_normal_dist(xRaw: float) -> float:
        if xRaw < 0:
            neg = 1
        else:
            neg = 0

        k = 1.0 / (1.0 + 0.2316419 * abs(xRaw))
        y = ((((1.330274429 * k - 1.821255978) * k + 1.781477937) * k - 0.356563782) * k + 0.319381530) * k
        y = 1.0 - 0.398942280401 * np.exp(-0.5 * (xRaw * xRaw)) * y
        return (1.0 - neg) * y + neg * (1.0 - y)

    @staticmethod
    def representation(sigma: float) -> float:
        pct = CombinedExperiment.cumulative_normal_dist(sigma)
        return 100 * (1 - 2 * abs(.5 - pct))

    @staticmethod
    def compute_winner_stats(results: List[Tuple[Candidate, List[Candidate]]]) -> (float, float):
        winners = list(map(lambda x: x[0].ideology.vec[0], results))
        sigmas = [abs(x) for x in winners]
        rr = [CombinedExperiment.representation(x) for x in winners]

        mean_sigma = np.mean(sigmas)
        mean_representation = np.mean(rr)
        return mean_sigma, mean_representation
