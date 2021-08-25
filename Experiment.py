import math
import os.path as path
from typing import List
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

from ExperimentConfig import ExperimentConfig
from elections.Candidate import Candidate
from elections.HeadToHeadElection import HeadToHeadElection
from elections.Ideology import Ideology
from elections.Voter import Voter
from network.ElectionMemory import ElectionMemory
from network.ElectionModel import ElectionModel, ElectionModelTrainer
from network.ResultMemory import ResultMemory


class RaceResult:
    def __init__(self, winner: Candidate, candidates: List[Candidate], base_candidates: List[Candidate],
                 utilities: np.array, base_utilities: np.array, condorcet_tie: bool = False):
        self.winner = winner
        self.candidates = candidates
        self.base_candidates = base_candidates
        self.utilities = utilities
        self.base_utilities = base_utilities
        self.condorcet_tie = condorcet_tie
        self.winner_expected_utility = 1 - abs(winner.ideology.distance_from_o())
        self.winner_idx = candidates.index(winner)
        self.winner_base = self.base_candidates[self.winner_idx]


class ErrorTracker:
    def __init__(self, decay: float, epsilon: float, min_iterations: int, max_static_iterations: int):
        self.decay = decay
        self.epsilon = epsilon
        self.min_iterations = min_iterations
        self.max_static_iterations = max_static_iterations
        self.last_step_iter = 0
        self.last_step_loss = 0
        self.iteration = 0
        self.loss = 0

    def add_loss(self, loss: float):
        # ignore the first 100 iterations, they are too noisy.
        if self.iteration == 0:
            self.loss = loss
            self.last_step_iter = self.iteration
            self.last_step_loss = loss
        else:
            self.loss = (1 - self.decay) * self.loss + self.decay * loss

            if self.loss < self.last_step_loss - self.epsilon:
                self.last_step_iter = self.iteration
                self.last_step_loss = loss

        self.iteration += 1

        return self.loss

    def complete(self) -> bool:
        return self.iteration > self.min_iterations and \
               self.iteration - self.last_step_iter > self.max_static_iterations


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

        # don't change anything of all options have zero return
        best_ideology = self.current_candidate.ideology.vec[0]
        best_return = self.win_bonus(best_ideology) * win_probabilities[0, self.current_bin]

        b_start = max(0, self.current_bin - bin_range)
        b_end = min(self.config.n_bins, self.current_bin + bin_range + 1)
        for b in range(b_start, b_end):
            ideology = self.config.convert_bin_to_ideology(b)
            bin_start = self.config.convert_bin_to_ideology_base(b)
            wb = self.win_bonus(ideology)
            expected_return = wb * win_probabilities[0, b]
            # print(f"%s: bin %2d bin_start: % 5.2f ideology % 6.2f  win_probability %5.3f win_bonus %5.3f return %5.3f" %
            #       (self.base_candidate.name, b, bin_start, ideology, win_probabilities[0, b], wb, expected_return), end = '')
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


class MemoryWrapper:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.result_memory = config.result_memory
        self.e_memory_path = "memory/%s-%s-%.2f.election_memory" % (self.config.election_name, self.config.equal_pct_bins, self.config.quality_variance)
        self.r_memory_path = "memory/%s-%.2f.result_memory" % (self.config.election_name, self.config.quality_variance)
        self.count = 0
        if self.result_memory:
            self.r_memory = ResultMemory(config.memory_size)
        else:
            self.e_memory = ElectionMemory(config.memory_size, config.n_bins)

    def save(self):
        if self.result_memory:
            with open(self.r_memory_path, "wb") as f:
                pickle.dump(self.r_memory, f)
            print("saved %d results to %s" % (self.count, self.r_memory_path))
        else:
            with open(self.e_memory_path, "wb") as f:
                pickle.dump(self.e_memory, f)
            print("saved %d results to %s" % (self.count, self.e_memory_path))

    def load(self) -> bool:
        if self.result_memory:
            if path.exists(self.r_memory_path):
                with open(self.r_memory_path, "rb") as f:
                    self.r_memory = pickle.load(f)
                self.count = self.r_memory.count
                return True
            else:
                return False
        else:
            if path.exists(self.e_memory_path):
                with open(self.e_memory_path, "rb") as f:
                    self.e_memory = pickle.load(f)
                self.count = self.e_memory.count
                return True
            else:
                return False

    def add_sample(self, result: np.ndarray):
        if self.result_memory:
            self.r_memory.add_sample(result)
            self.count = self.r_memory.count
        else:
            cc = [self.config.convert_ideology_to_bin(i) for i in result]
            self.e_memory.add_sample(cc, 0)
            self.count = self.e_memory.count

    def get_batch(self) -> (np.ndarray, np.ndarray, np.ndarray):
        if self.result_memory:
            return self.config.create_batch_from_results(self.r_memory)
        else:
            return self.e_memory.get_batch(self.config.batch_size)


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._model = None
        self.trainer = None
        self.model_path = self.config.model_path

        self.loss_tracker = ErrorTracker(decay=1e-3,
                                         epsilon=1e-5,
                                         min_iterations=self.config.training_cycles,
                                         max_static_iterations=20000)

        self.memory = MemoryWrapper(self.config)
        self.training_count = 0

    def model(self) -> ElectionModel:
        if self._model:
            return self._model

        return self.get_or_train_model()

    def get_or_train_model(self) -> ElectionModel:
        if path.exists(self.model_path):
            self._model = tf.keras.models.load_model(self.model_path, compile=False)
        elif self.config.build_model:
            print(f"training model:  {self.model_path}")
            self._model = self.train_model()
            self._model.save(self.model_path)
        else:
            raise Exception("Model (%s) not prebuilt and config.build_model is False." % self.model_path)

        return self._model

    def train_model(self) -> ElectionModel:
        network = ElectionModel(self.config.name, self.config.n_bins, self.config.model_width, self.config.model_layers, self.config.sigmoid)
        self.trainer = ElectionModelTrainer(network, self.config)
        self.populate_memory()
        network, loss = self.train_network(network, self.config.training_cycles, self.config.batch_size)
        return network

    def populate_memory_par(self, count: int):
        print("populate_memory_par for %s" % self.config.name)
        results: List[np.ndarray] = Parallel(n_jobs=32)(
            delayed(self.config.create_sample_for_memory)() for _ in range(count))
        for r in results:
            self.memory.add_sample(r)
        print("populate_memory_par complete:  size %d" % self.memory.count)

    def populate_memory(self):
        if self.memory.load():
            print("loaded %d results from memory" % self.memory.count)
        else:
            self.populate_memory_par(self.config.memory_size)
            self.memory.save()

    def compute_random_results_par(self, count: int) -> List[RaceResult]:
        results: List[RaceResult] = Parallel(n_jobs=32)(delayed(self.run_random_race)() for _ in range(count))
        return results

    def run_random_race(self) -> RaceResult:
        candidates = self.config.gen_candidates(5)
        return self.run_random_race_c(candidates)

    def run_random_race_c(self, candidates: List[Candidate]) -> RaceResult:
        voters = self.config.population.generate_unit_voters(self.config.sampling_voters)
        result = self.config.run_election(candidates, voters)
        winner = result.winner()
        utilities = self.compute_utilities(winner, candidates, voters)
        return RaceResult(winner, candidates, candidates, utilities, utilities, result.is_tie)

    def compute_random_results(self, count: int) -> List[RaceResult]:
        results: List[RaceResult] = []
        for i in range(count):
            results.append(self.run_random_race())
        return results

    @staticmethod
    def compute_utilities(winner: Candidate, all_candidates: List[Candidate], voters: List[Voter]) -> np.ndarray:
        c_set = set(all_candidates)
        c_set.remove(winner)
        candidates = [winner] + list(c_set)

        def compute_utility(c) -> float:
            u = [-v.ideology.distance(c.ideology) for v in voters]
            return np.mean(u)

        return np.array([compute_utility(c) for c in candidates])

    def train_network(self, net: ElectionModel, n_batches: int, batch_size: int):
        print(f"training network for max of {n_batches} epochs")
        tracker = self.loss_tracker
        i = self.training_count
        average_loss = 0
        end = self.training_count + n_batches
        report = 10
        while i < end and not tracker.complete():
            i += 1
            # x, a, y = self.config.create_batch_from_results(self.memory)
            x, a, y = self.memory.get_batch()
            loss = self.trainer.update(x, a, y)
            if np.isnan(loss):
                raise Exception("loss is nan")

            average_loss = tracker.add_loss(loss)
            if i % report == 0:
                if report < 1000:
                    report = report * 10
                # print(f"Epoch {i:5d} loss = {average_loss:.6}")
                print("%s Epoch %5d loss = %.6f" % (self.config.name, i, average_loss))
                progress_path = "%s.%06d.progress" % (self.model_path, i)
                net.save(progress_path, overwrite=True)
                net.save(self.model_path, overwrite=True)

        return net, average_loss

    @staticmethod
    def log_candidates(msg: str, cc: List[Candidate]):
        print(msg)
        for c in cc:
            print(f"\tcandidate %s % 6.2f" % (c.name, c.ideology.vec[0]))

    def run_strategic_races(self, n: int) -> List[RaceResult]:
        results: List[RaceResult] = []
        for i in range(n):
            results.append(self.run_strategic_race())
        return results

    def run_strategic_races_par(self, n: int) -> List[RaceResult]:
        winners: List[RaceResult] = Parallel(n_jobs=32)(delayed(self.run_strategic_race)() for _ in range(n))
        return winners

    def run_strategic_race(self) -> RaceResult:
        candidates = self.config.gen_candidates(5)
        voters = self.config.population.generate_unit_voters(self.config.sampling_voters)
        return self.run_strategic_race_c(candidates, voters)

    def run_strategic_race_c(self, candidates: List[Candidate], voters: List[Voter]) -> RaceResult:
        model = self.model()
        base_candidates = candidates
        # self.log_candidates(f"starting candidates {self.config.election_name}", candidates)
        extended_candidates = [ExtendedCandidate(c, self.config) for c in candidates]
        for bin_range in [3, 2, 1]:
            cc = [ec.current_candidate for ec in extended_candidates]
            candidates = [e.best_candidate(model, cc, bin_range) for e in extended_candidates]
            # self.log_candidates(f"after adjust {bin_range}", candidates)

        HeadToHeadElection.count_of_ties = 0
        result = self.config.run_election(candidates, voters)
        w = result.winner()
        strategic_utilities = self.compute_utilities(w, candidates, voters)
        base_utilities = self.compute_utilities(base_candidates[0], base_candidates, voters)
        return RaceResult(w, candidates, base_candidates, strategic_utilities, base_utilities, result.is_tie)

    @staticmethod
    def plot_results(results: List[List[float]], title: str, labels: List[str]):
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
        axis.tick_params(axis='y', colors="black")
        axis.set_xlim([-1, 1])

        bins = np.arange(-1, 1, 2 / 21)
        axis.hist(results, bins=bins, label=labels, edgecolor='white', stacked=True)
        axis.legend()
        axis.set_xlabel("Sigma From Median Voter", fontsize=20)
        axis.set_ylabel("Frequency of Winner at Ideology", fontsize=20)

        plt.savefig("foo.png")
