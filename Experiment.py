import math
import os.path as path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

from ExperimentConfig import ExperimentConfig
from elections.Ballot import Ballot
from elections.Candidate import Candidate
from elections.DefaultConfigOptions import unit_election_config
from elections.Ideology import Ideology
from network.ElectionMemory import ElectionMemory
from network.ElectionModel import ElectionModel, ElectionModelTrainer
from network.LossTracker import LossTracker


class ExtendedCandidate:
    def __init__(self, base_candidate: Candidate, config: ExperimentConfig):
        self.config = config
        self.base_candidate = base_candidate
        self.current_bin = self.config.convert_ideology_to_bin(base_candidate.ideology.vec[0])
        self.current_candidate = base_candidate

    def win_bonus(self, ideology: float) -> float:
        delta = math.fabs(ideology - self.base_candidate.ideology.vec[0])
        wb = max(1 - delta / max(1e-5, self.config.ideology_flexibility), 0.0)
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
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._model = None
        self.memory = None

        self.model_path = f"{self.config.model_path}.mdl"

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
        network, loss = self.train_network(network, self.memory, self.config.training_cycles, self.config.batch_size)
        return network

    def populate_memory(self, count: int) -> ElectionMemory:
        m = ElectionMemory(count * 10, self.config.n_bins)
        print(f"populating training memory with {count * 10} samples")
        for i in range(count):
            cc = self.config.gen_random_candidates(5)
            w = self.run_sample_election(cc, self.config.training_voters)
            ci, wi = self.config.create_training_sample(cc, w)
            m.add_sample(ci, wi)
            if i % 1000 == 0:
                print(f"sample {i}", end='\r')
        print(f"\nm.count {m.count}")
        self.memory = m
        return self.memory

    def compute_random_results(self, count: int) -> List[Tuple[Candidate, List[Candidate]]]:
        results = []
        for i in range(count):
            candidates = self.config.gen_candidates(5)
            winner = self.run_sample_election(candidates, self.config.sampling_voters)
            results.append((winner, candidates))
        return results

    def run_sample_election(self,
                            candidates: List[Candidate],
                            n_voters: int) -> Candidate:
        voters = self.config.population.generate_unit_voters(n_voters)
        ballots = [Ballot(v, candidates, unit_election_config) for v in voters]
        process = self.config.election_constructor()
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
                    current_path = f"{self.config.model_path}.{i}"
                    net.save(current_path, overwrite=True)

        return net, average_loss

    @staticmethod
    def log_candidates(msg: str, cc: List[Candidate]):
        print(msg)
        for c in cc:
            print(f"\tcandidate {c.name}, {c.ideology.vec[0]:.4f}")

    def run_strategic_races_core(self, n: int) -> List[Tuple[Candidate, List[Candidate]]]:
        model = self.model()
        winners = []
        for i in range(n):
            w, cc = self.run_strategic_race()
            winners.append((w, cc))

        return winners

    def run_strategic_races_par(self, n: int) -> List[Tuple[Candidate, List[Candidate]]]:
        model = self.model()
        winners = Parallel(n_jobs=16)(delayed(self.run_strategic_race)() for _ in range(n))
        return winners

    def run_strategic_race(self) -> (Candidate, List[Candidate]):
        model = self.model()
        candidates = self.config.gen_candidates(5)
        extended_candidates = [ExtendedCandidate(c, self.config) for c in candidates]
        for bin_range in [3, 2, 1]:
            cc = [ec.current_candidate for ec in extended_candidates]
            candidates = [e.best_candidate(model, cc, bin_range) for e in extended_candidates]
            # self.log_candidates(f"after adjust {bin_range}", candidates)

        w = self.run_sample_election(candidates, self.config.sampling_voters)
        return w, cc

    @staticmethod
    def plot_results(results: List[List[float]], title: str, labels: List[str]):
        n_rows = 1
        n_cols = 1
        fig, axis = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))
        fig.suptitle(title, color="black", fontsize=22)
        fig.set_facecolor("white")

        count = 0
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        axis.tick_params(axis='x', colors="black")
        axis.tick_params(axis='y', colors="black")
        axis.set_xlim([-1, 1])

        bins = np.arange(-1, 1, 2 / 21)
        axis.hist(results, bins=bins, label=labels, edgecolor='white', stacked=True)
        axis.legend()
        axis.set_xlabel("Sigma From Origin", fontsize=20)
        axis.set_ylabel("Frequency of Winner at Ideology", fontsize=20)

        plt.savefig("foo.png")
