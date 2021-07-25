from elections.Ideology import Ideology
import random
from typing import List

import numpy as np

import matplotlib.pyplot as plt
from elections.Ballot import Ballot
from elections.Candidate import Candidate
from elections.Ideology import Ideology
from elections.NDPopulation import NDPopulation
from elections.PopulationGroup import Independents
from elections.ElectionConstructor import ElectionConstructor
from network.Tensor import Tensor
from network.ElectionModel import ElectionModel
from ExperimentalConfig import ExperimentalConfig
from network.ElectionMemory import ElectionMemory
from network.LossTracker import LossTracker
from elections.DefaultConfigOptions import unit_election_config
import os.path as path
import tensorflow as tf
from network.ElectionModel import ElectionModel, ElectionModelTrainer
from network.ElectionMemory import ElectionMemory
import math


class ExtendedCandidate:
    def __init__(self, base_candidate: Candidate, config: ExperimentalConfig):
        self.config = config
        self.base_candidate = base_candidate
        self.current_bin = self.config.convert_ideology_to_bin(base_candidate.ideology.vec[0])
        self.current_candidate = base_candidate

    def win_bonus(self, ideology: float) -> float:
        delta = math.fabs(ideology - self.base_candidate.ideology.vec[0])
        wb = max(1 - delta / self.config.ideology_flexibility, 0.0)
        # print(f"win_bonus:  base_ideology {self.base_candidate.ideology.vec[0]:.4f}" +
        #       f"ideology {ideology:.4f} delta {delta:.4f}")
        return wb

    def best_position(self, model: ElectionModel, other_candidates: List[Candidate], bin_range: int) -> float:
        x = self.config.convert_candidates_to_input_vec(other_candidates)
        win_probabilities = model(x).numpy()
        # print("win_probabilities")
        # print(win_probabilities)

        # don't change anything of all options have zero return
        best_return = 1e-6
        best_ideology = self.current_candidate.ideology.vec[0]

        b_start = self.current_bin - bin_range
        b_end = min(self.config.n_bins, self.current_bin + bin_range + 1)
        for b in range(b_start, b_end):
            ideology = self.config.convert_bin_to_ideology(b)
            wb = self.win_bonus(ideology)
            expected_return = wb * win_probabilities[0, b]
            # print(f"bin {b:2d} ideology {ideology: .4f} win_probability {win_probabilities[0, b]:.4f} " +
            #       f"win_bonus {wb:.4f} expected_return {expected_return:.4f}", end='')
            if expected_return > best_return:
                # print(" best!", end='')
                best_return = expected_return
                best_ideology = ideology
            # print('')

        return best_ideology

    def best_candidate(self, model: ElectionModel, all_candidates: List[Candidate], bin_range: int):
        other_candidates = list(filter(lambda c: c != self.current_candidate, all_candidates))
        assert (len(other_candidates) == 4)
        ideology = self.best_position(model, other_candidates, bin_range)
        self.current_bin = self.config.convert_ideology_to_bin(ideology)

        self.current_candidate = Candidate(
            self.base_candidate.name,
            self.base_candidate.party,
            Ideology(np.array([ideology])),
            self.base_candidate.quality)

        return self.current_candidate


class Experiment:
    def __init__(self, config: ExperimentalConfig):
        self.config = config
        self._model = None
        self.memory = None

        self.model_path = f"{self.config.path}.mdl"

    def model(self) -> ElectionModel:
        if self._model:
            return self._model

        return self.get_or_train_model()

    def get_or_train_model(self) -> ElectionModel:
        if path.exists(self.model_path):
            # print(f"loading {self.model_path}")
            self._model = tf.keras.models.load_model(self.model_path)
        else:
            self._model = self.train_model()
            self._model.save(self.model_path)

        return self._model

    def train_model(self) -> ElectionModel:
        network = ElectionModel(self.config.n_bins, self.config.model_width, self.config.model_layers)
        self.populate_memory(self.config.memory_size)
        network, loss = self.train_network(network, self.memory, 10000, self.config.batch_size)
        return network

    def populate_memory(self, count: int) -> ElectionMemory:
        m = ElectionMemory(count * 5, self.config.n_bins)
        process = self.config.election_constructor
        print(f"populating training memory with {count * 5} samples")
        for i in range(count):
            cc = self.config.gen_random_candidates(5)
            w = self.run_sample_election(cc, process, self.config.training_voters)
            ci, wi = self.config.create_training_sample(cc, w)
            m.add_sample(ci, wi)
            if i % 100 == 0:
                print(".", end='')
        print(f"\nm.count {m.count}")
        self.memory = m
        return self.memory

    def run_sample_election(self, candidates: List[Candidate],
                            process: ElectionConstructor,
                            n_voters: int) -> Candidate:
        voters = self.config.population.generate_unit_voters(n_voters)
        ballots = [Ballot(v, candidates, unit_election_config) for v in voters]
        result = process.run(ballots, set(candidates))
        winner = result.winner()
        return winner

    def train_network(self, net: ElectionModel, memory: ElectionMemory, n_batches: int, batch_size: int):
        print(f"training network for {n_batches} epochs")
        tracker = LossTracker(1000)
        trainer = ElectionModelTrainer(net)
        current_path = ""
        average_loss = 0

        for i in range(n_batches):
            x, a, y = memory.get_batch(batch_size)
            loss = trainer.update(x, a, y)
            if np.isnan(loss):
                print(f"loss is nan, reverting to {current_path}")
                net = tf.keras.models.load_model(current_path)
                trainer = ElectionModelTrainer(net)
            else:
                average_loss = tracker.add_loss(loss)
                if i % 1000 == 0:
                    print(f"epoch {i:5d} loss = {average_loss:.6}")
                    current_path = f"mdl.sav.{i}"
                    net.save(current_path, overwrite=True)

        return net, average_loss

    def plot_results(self, results: List[List[float]], labels: List[str]):
        n_rows = 1
        n_cols = 1
        fig, axis = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))
        fig.suptitle("Ideology of Winner with Strategic Candidates", color="black", fontsize=22)
        fig.set_facecolor("white")

        count = 0

        axis.tick_params(axis='x', colors="black")
        axis.tick_params(axis='y', colors="black")
        # axis.set_xlim([0, 2])

        axis.hist(results, bins=30, label=labels)
        axis.legend()

        plt.savefig("foo.png")

    @staticmethod
    def log_candidates(msg: str, cc: List[Candidate]):
        print(msg)
        for c in cc:
            print(f"\tcandidate {c.name}, {c.ideology.vec[0]:.4f}")

    def run_strategic_races(self, n: int) -> np.ndarray:
        process = self.config.election_constructor
        model = self.model()
        population = self.config.population

        winners = []
        for i in range(n):
            candidates = self.config.gen_candidates(5)
            # self.log_candidates(f"starting candidates", candidates)
            extended_candidates = [ExtendedCandidate(c, self.config) for c in candidates]
            for bin_range in [3, 2, 1]:
                cc = [ec.current_candidate for ec in extended_candidates]
                candidates = [e.best_candidate(model, cc, bin_range) for e in extended_candidates]
                # self.log_candidates(f"after adjust {bin_range}", candidates)

            w = self.run_sample_election(candidates, process, self.config.sampling_voters)
            if i % 100 == 0:
                print(f"{i:5d} w.ideology: {w.ideology.vec[0]:.4}")

            winners.append(w.ideology.vec[0])

        return np.array(winners)
