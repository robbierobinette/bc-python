from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import os
from Experiment import Experiment
from ExperimentConfig import ExperimentConfig
from elections.Candidate import Candidate
from elections.HeadToHeadElection import HeadToHeadElection


class CombinedExperiment:
    def __init__(self, configs: List[ExperimentConfig], path_base: str, n_races: int):
        self.path_base = path_base
        self.configs = configs
        self.n_races = n_races
        self.names = ["c-0", "c-1", "c-2", "c-3", "c-4"]

        self.experiments = [Experiment(config) for config in configs]

    def run(self):
        for exp in self.experiments:
            self.run_experiement(exp)

    def run_experiement(self, exp: Experiment):
        HeadToHeadElection.count_of_ties = 0
        strategic_results = exp.run_strategic_races_core(self.n_races)
        if exp.config.election_name == "H2H":
            print(f"number of Condorcet cycles:  {HeadToHeadElection.count_of_ties}")
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

        print("%14s random sigma %5.2f score %5.2f strategic sigma %5.2f score %5.2f" %
              (name, r_sigma, r_score, s_sigma, s_score))

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

    def compute_winner_stats(self, results: List[Tuple[Candidate, List[Candidate]]]) -> (float, float):
        winners = list(map(lambda x: x[0].ideology.vec[0], results))
        sigmas = [abs(x) for x in winners]
        rr = [CombinedExperiment.representation(x) for x in winners]

        mean_sigma = np.mean(sigmas)
        mean_representation = np.mean(rr)
        return mean_sigma, mean_representation
