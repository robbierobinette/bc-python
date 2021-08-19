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
        best_return = 1e-6
        best_ideology = self.current_candidate.ideology.vec[0]

        b_start = max(0, self.current_bin - bin_range)
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
        self.trainer = None
        self.model_path = f"{self.config.model_path}.mdl"

        self.error_tracker = ErrorTracker(decay=1e-3,
                                          epsilon=1e-5,
                                          min_iterations=self.config.training_cycles,
                                          max_static_iterations=20000)

        self.memory = ResultMemory(self.config.memory_size)
        self.training_count = 0
        self.memory_path = "memory/%s.results" % config.election_name

    def model(self) -> ElectionModel:
        if self._model:
            return self._model

        return self.get_or_train_model()

    def get_or_train_model(self) -> ElectionModel:
        if path.exists(self.model_path):
            print(f"loading model: {self.model_path}")
            self._model = tf.keras.models.load_model(self.model_path, compile=False)
        else:
            print(f"training model:  {self.model_path}")
            self._model = self.train_model()
            self._model.save(self.model_path)

        return self._model

    def train_model(self) -> ElectionModel:
        network = ElectionModel(self.config.name, self.config.n_bins, self.config.model_width, self.config.model_layers)
        self.trainer = ElectionModelTrainer(network, self.config)
        self.populate_memory()
        network, loss = self.train_network(network, self.config.training_cycles, self.config.batch_size)
        return network

    def populate_memory_strategic(self, count: int):
        for i in range(count):
            r = self.run_strategic_race()
            result_vec = self.config.create_training_sample(r.candidates, r.winner)
            self.memory.add_sample(result_vec)

    def populate_memory_par(self, count: int):
        print("populate_memory_par for %s" % self.model_path)
        results: List[np.ndarray] = Parallel(n_jobs=32)(
            delayed(self.config.create_sample_for_memory)() for _ in range(count))
        for r in results:
            self.memory.add_sample(r)
        print("populate_memory_par complete:  size %d" % self.memory.count)

    def populate_memory(self):
        if path.exists(self.memory_path):
            with open(self.memory_path, "rb") as f:
                self.memory = pickle.load(f)
            print("loaded %d results from %s" % (self.memory.count, self.memory_path))
        else:
            self.populate_memory_par(self.config.memory_size)
            with open(self.memory_path, "wb") as f:
                pickle.dump(self.memory, f)
            print("saved %d results to %s" % (self.memory.count, self.memory_path))

    def populate_memory_serial(self, count: int):
        print(f"populating training memory with {count * 10} samples")
        for i in range(count):
            ci, wi = self.config.create_sample_for_memory()
            self.memory.add_sample(ci, wi)

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
        tracker = self.error_tracker
        progress_path = f"{self.config.model_path}.progress"
        average_loss = 0

        i = self.training_count
        end = self.training_count + n_batches
        report = 10
        while i < end and not tracker.complete():
            i += 1
            x, a, y = self.config.create_batch_from_results(self.memory)
            loss = self.trainer.update(x, a, y)
            assert(not np.isnan(loss), "loss is nan")

            average_loss = tracker.add_loss(loss)
            if i % report == 0:
                if report < 1000:
                    report = report * 10
                print(f"Epoch {i:5d} loss = {average_loss:.6}")
                net.save(progress_path, overwrite=True)

        return net, average_loss

    @staticmethod
    def log_candidates(msg: str, cc: List[Candidate]):
        print(msg)
        for c in cc:
            print(f"\tcandidate {c.name}, {c.ideology.vec[0]:.4f}")

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
        return self.run_strategic_race_c(candidates)

    def run_strategic_race_c(self, candidates: List[Candidate]) -> RaceResult:
        model = self.model()
        base_candidates = candidates
        extended_candidates = [ExtendedCandidate(c, self.config) for c in candidates]
        for bin_range in [3, 2, 1]:
            cc = [ec.current_candidate for ec in extended_candidates]
            candidates = [e.best_candidate(model, cc, bin_range) for e in extended_candidates]
            # self.log_candidates(f"after adjust {bin_range}", candidates)

        voters = self.config.population.generate_unit_voters(self.config.sampling_voters)
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
        axis.set_xlabel("Sigma From Origin", fontsize=20)
        axis.set_ylabel("Frequency of Winner at Ideology", fontsize=20)

        plt.savefig("foo.png")
