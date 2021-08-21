from CombinedExperiment import CombinedExperiment
from ExperimentConfig import ExperimentConfig
from Experiment import Experiment
from util.snap import snap
from copy import copy
from typing import List
from joblib import Parallel, delayed
from CombinedExperiment import ExperimentResult
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "5"

version = "v26"
snap(version)
n_races = 1000
base_config = ExperimentConfig(name="none",
                               election_name="none",
                               training_cycles=50000,
                               ideology_range=1.5,
                               ideology_flexibility=.7,
                               n_bins=21,
                               model_width=768,
                               model_layers=4,
                               memory_size=200000,
                               result_memory=True,
                               batch_size=2048,
                               training_voters=1000,
                               sampling_voters=1000,
                               quality_variance=.1,
                               candidate_variance=0.5,
                               equal_pct_bins=True,
                               model_path="none")

# irv_x_config = copy(base_config)
# irv_x_config.name = "IRV-X"
# irv_x_config.election_name = "IRV"
# irv_x_config.model_path = f"exp/{version}/IRV-X"
# irv_x_config.equal_pct_bins = True
# irv_x_config.result_memory = True
# irv_x_config.build_model = True
#
# irv_y_config = copy(base_config)
# irv_y_config.name = "IRV-Y"
# irv_y_config.election_name = "IRV"
# irv_y_config.model_path = f"exp/{version}/IRV-Y"
# irv_y_config.equal_pct_bins = True
# irv_y_config.result_memory = False
# irv_y_config.build_model = True
# base_configs = [irv_x_config, irv_y_config]


irv_a_config = copy(base_config)
irv_a_config.name = "IRV-A"
irv_a_config.election_name = "IRV"
irv_a_config.model_path = f"exp/{version}/IRV-A.mdl"
irv_a_config.equal_pct_bins = True

h2h_a_config = copy(base_config)
h2h_a_config.name = "H2H-A"
h2h_a_config.election_name = "H2H"
h2h_a_config.model_path = f"exp/{version}/H2H-A.mdl"
h2h_a_config.equal_pct_bins = True


irv_b_config = copy(base_config)
irv_b_config.name = "IRV-B"
irv_b_config.election_name = "IRV"
irv_b_config.model_path = f"exp/{version}/IRV-B.mdl"
irv_b_config.equal_pct_bins = False

h2h_b_config = copy(base_config)
h2h_b_config.name = "H2H-B"
h2h_b_config.election_name = "H2H"
h2h_b_config.model_path = f"exp/{version}/H2H-B.mdl"
h2h_b_config.equal_pct_bins = False

base_configs = [
    h2h_a_config, irv_a_config,
    h2h_b_config, irv_b_config,
]


def touch_model(config: ExperimentConfig):
    config.build_model = True
    x = Experiment(config).model()
    return True


def build_base_models(configs: List[ExperimentConfig]):
    for v in configs:
        exp = Experiment(v)
        print("populating memory for %s " % exp.config.name)
        exp.populate_memory()

    # base_results = [touch_model(c) for c in configs]
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
    exp = CombinedExperiment([config], path_base=f"exp/{version}", n_races=n_races)
    result = exp.run()[0]
    return result


def build_variants():
    # q_v = build_quality_variants([h2h_a_config])
    # f_v = build_flex_variants(base_configs)
    all_variants = base_configs
    print(f"{len(all_variants)} variants to build.")

    # all_results = Parallel(n_jobs=32)(delayed(run_variant)(c) for c in all_variants)
    all_results = [run_variant(c) for c in all_variants]

    import os
    os.system(f"mkdir -p exp/{version}")
    with open(f"exp/{version}/summary.txt", "w") as f:
        for r in all_results:
            f.write(r.to_string() + "\n")

    for r in all_results:
        r.print()


base_results = build_base_models(base_configs)
# print(f"base_results: {base_results}")
# build_variants()
