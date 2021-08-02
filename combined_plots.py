from CombinedExperiment import CombinedExperiment
from ExperimentConfig import ExperimentConfig
from util.snap import snap

snap("v1")
n_races = 1000
irv_config = ExperimentConfig("IRV",
                              training_cycles = 20000,
                              ideology_range = 1.5,
                              ideology_flexibility = .7,
                              n_bins = 21,
                              model_width = 512,
                              model_layers = 3,
                              memory_size = 50000,
                              batch_size = 2048,
                              training_voters = 400,
                              sampling_voters = 1000,
                              quality_variance = .1,
                              path = "exp/h2h-0")

h2h_config = ExperimentConfig("H2H",
                              training_cycles = 20000,
                              ideology_range = 1.5,
                              ideology_flexibility = .7,
                              n_bins = 21,
                              model_width = 512,
                              model_layers = 3,
                              memory_size = 50000,
                              batch_size = 2048,
                              training_voters = 400,
                              sampling_voters = 1000,
                              quality_variance =.1,
                              path = "exp/h2h-0")

pty_config = ExperimentConfig("Plurality",
                              training_cycles = 20000,
                              ideology_range = 1.5,
                              ideology_flexibility = .7,
                              n_bins = 21,
                              model_width = 512,
                              model_layers = 3,
                              memory_size = 50000,
                              batch_size = 2048,
                              training_voters = 400,
                              sampling_voters = 1000,
                              quality_variance = .1,
                              path = "exp/h2h-0")

cexp = CombinedExperiment([h2h_config, irv_config, pty_config], "exp/v1", 1000)
cexp.run()
