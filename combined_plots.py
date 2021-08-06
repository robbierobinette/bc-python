from CombinedExperiment import CombinedExperiment
from ExperimentConfig import ExperimentConfig
from Experiment import Experiment
from util.snap import snap
from copy import copy
from typing import List
from joblib import Parallel, delayed
from CombinedExperiment import ExperimentResult

version = "v6"
snap(version)
n_races = 1000
base_config = ExperimentConfig(name="none",
                               election_name="none",
                               training_cycles=20000,
                               ideology_range=1.5,
                               ideology_flexibility=.7,
                               n_bins=21,
                               model_width=512,
                               model_layers=3,
                               memory_size=100000,
                               batch_size=2048,
                               training_voters=400,
                               sampling_voters=1000,
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

pty_config = copy(base_config)
pty_config.election_name = "Plurality"
pty_config.model_path = f"exp/{version}/plurality"
pty_config.name = "Plurality"

base_configs = [h2h_config, irv_config, pty_config]


def touch_model(config: ExperimentConfig):
    x = Experiment(config).model()
    return True


def build_base_models(configs: List[ExperimentConfig]):
    base_results = Parallel(n_jobs=16)(delayed(touch_model)(c) for c in configs)
    return base_results


import numpy as np


def build_quality_variants(base_configs: List[ExperimentConfig]) -> List[ExperimentConfig]:
    cc: List[ExperimentConfig] = []
    for c in base_configs:
        for qv in np.arange(0, .21, .02):
            nc = copy(c)
            nc.quality_variance = qv
            nc.name = "%s-qv-%04.02f" % (c.name, qv)
            cc.append(nc)
    return cc


def build_flex_variants(base_configs: List[ExperimentConfig]) -> List[ExperimentConfig]:
    cc: List[ExperimentConfig] = []
    for c in base_configs:
        for fx in np.arange(0, 1.01, .1):
            nc = copy(c)
            nc.ideology_flexibility = fx
            nc.name = "%s-flex-%04.1f" % (c.name, fx)
            cc.append(nc)
    return cc


def run_variant(config: ExperimentConfig) -> ExperimentResult:
    exp = CombinedExperiment([config], path_base="exp/v0", n_races=n_races)
    result = exp.run()[0]
    return result


def build_variants():
    q_v = build_quality_variants([h2h_config])
    f_v = build_flex_variants(base_configs)
    all_variants = q_v + f_v
    print(f"{len(all_variants)} variants to build.")

    all_results = Parallel(n_jobs=32)(delayed(run_variant)(c) for c in all_variants)
    # all_results = [run_variant(c) for c in all_variants]

    import os
    os.system(f"mkdir -p exp/{version}")
    with open(f"exp/{version}/summary.txt", "w") as f:
        for r in all_results:
            f.write(r.to_string() + "\n")

    for r in all_results:
        r.print()


base_results = build_base_models(base_configs)
print(f"base_results: {base_results}")
build_variants()
