import os
from util.Constructor import Constructor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "5"
from copy import copy
import numpy as np

from joblib import Parallel, delayed

from ExperimentConfig import ExperimentConfig
from Experiment import Experiment
from CombinedExperiment import CombinedExperiment
from typing import List
from Experiment import RaceResult


class ComparisonResult:
    def __init__(self, results: List[RaceResult]):
        self.results = results


version="v29"

base_config = ExperimentConfig("base_config",
                               "IRV",
                               equal_pct_bins=True,
                               candidate_variance=.5,
                               quality_variance=0,
                               ideology_flexibility=.7,
                               sampling_voters=1000,
                               model_path=f"exp/{version}/IRV-A.mdl.010000.progress")


def run_strategic_races(config: ExperimentConfig) -> RaceResult:
    candidates = base_config.gen_candidates(5)
    voters = config.population.generate_unit_voters(config.sampling_voters)
    x = Experiment(config)
    return x.run_strategic_race_c(candidates, voters)


def compute_SUE_single(rr: RaceResult):
    return CombinedExperiment.compute_SUE_single(rr)


import matplotlib.pyplot as plt


def make_line_plot(data, title, labels,
                   xlabel: str = "Candidate Ideological Flexibility (stddev)",
                   ylabel: str = "Social Utility Efficiency"):
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300
    n_rows = 1
    n_cols = 1
    fig, axis = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))
    fig.suptitle(title, color="black", fontsize=22)
    fig.set_facecolor("white")

    count = 0
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    axis.tick_params(axis='x', colors="black")

    for i in range(len(data)):
        d = data[i]
        plt.plot(d[0], d[1], label=labels[i])
    plt.legend()

    axis.set_xlabel(xlabel, fontsize=20)
    axis.set_ylabel(ylabel, fontsize=20)


def run_races(qv: float, flex: float):
    training_iterations = 50000
    h2h_config = copy(base_config)
    h2h_config.election_name = "H2H"
    h2h_config.model_path = f"exp/{version}/H2H.mdl"
    h2h_config.equal_pct_bins = False
    h2h_config.sampling_voters = 1000
    h2h_config.quality_variance = qv
    h2h_config.ideology_flexibility = flex

    n_races = 1000
    results: List[List[RaceResult]] = Parallel(n_jobs=32)(
        delayed(run_strategic_races)(h2h_config) for _ in range(n_races))
    return results


def count_ties(rr: List[RaceResult]) -> float:
    for r in rr:
        if r.condorcet_tie:
            print("tie")

    ties = [r for r in rr if r.condorcet_tie]
    return len(ties) / len(rr)



def main():
    flex_range = np.arange(0, 1.01, .1)
    qv_range = np.arange(0, .03, .02)

    flex_results_f = lambda: [run_races(0, fx) for fx in flex_range]
    qv_results_f = lambda: [run_races(qv, .7) for qv in qv_range]

    flex_results = Constructor(flex_results_f, f"exp/{version}/flex_results.p").construct()
    qv_results = Constructor(qv_results_f, f"exp/{version}/qv_results.p").construct()

    flex_ties = [count_ties(rr) for rr in flex_results]
    qv_ties = [count_ties(rr) for rr in qv_results]

    make_line_plot([[flex_range, flex_ties]],
                   "Frequency of Condorcet-Ties vs. Candidate Flexibility with Strategy",
                   ["Chance of Tie"],
                   "Candidate Ideological Flexibility (stddev)",
                   "Percentage of Ties")

    make_line_plot([[qv_range, qv_ties]],
                   "Frequency of Condorcet-Ties vs. Candidate Quality Variance",
                   ["Chance of Tie"],
                   "Quality Variance (Qc, stddev)",
                   "Percentage of Ties")

    plt.show()

main()