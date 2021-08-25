from CombinedExperiment import CombinedExperiment
from ExperimentConfig import ExperimentConfig
from Experiment import Experiment
from util.snap import snap
from copy import copy
from typing import List
from joblib import Parallel, delayed
from CombinedExperiment import ExperimentResult
import numpy as np
import random
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
                               sampling_voters=1000,
                               quality_variance=0,
                               candidate_variance=0.5,
                               equal_pct_bins=True,
                               model_path="none")


h2h_config = copy(base_config)
h2h_config.election_name = "H2H"
h2h_config.model_path = f"exp/{version}/H2H.mdl"
h2h_config.name = "Condorcet-Minimax"


exp = Experiment(h2h_config)
for seed in range(100):
    np.random.seed(seed)
    random.seed(seed)
    result = exp.run_strategic_race()
    print(f"seed {seed}: result.is_condorcet_tie {result.condorcet_tie}")
