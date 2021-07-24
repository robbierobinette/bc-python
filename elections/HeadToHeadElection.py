from typing import List, Set, Tuple

import numpy as np

from .Candidate import Candidate
from .Election import Election, BallotIter
from .ElectionResult import ElectionResult


class HeadToHeadResult(ElectionResult):
    def __init__(self, oc, rm):
        super().__init__(oc)
        self.rm = rm
    # def __init__(self, ordered_candidates: List[Candidate], result_matrix: npt.ArrayLike[float]):
    #     super().__init__(ordered_candidates)
    #     self.result_matrix = result_matrix


class HeadToHeadElection(Election):
    def __init__(self, ballots: BallotIter, candidates: Set[Candidate]):
        super().__init__(ballots, candidates)
        self.candidate_list = list(self.candidates)
        self.indices = {}
        for i in range(len(self.candidate_list)):
            self.indices[self.candidate_list[i]] = i

        self.result_matrix = self.compute_matrix()

    def result(self) -> ElectionResult:
        oc = self.minimax(self.candidates)
        return HeadToHeadResult(oc, self.result_matrix)

    def compute_matrix(self) -> np.array:
        n_candidates = len(self.candidates)
        results = np.zeros([n_candidates, n_candidates])
        for b in self.ballots:
            not_seen: Set[Candidate] = self.candidates.copy()
            for cs1 in b.ordered_candidates:
                c1 = cs1.candidate
                not_seen.remove(c1)
                row_i = self.indices[c1]
                for c2 in not_seen:
                    col_i = self.indices[c2]
                    results[row_i, col_i] += 1

        return results

    def delta(self, c1: Candidate, c2: Candidate) -> float:
        r = self.indices[c1]
        c = self.indices[c2]
        return self.result_matrix[r, c] - self.result_matrix[c, r]

    def max_loss(self, candidate: Candidate, active_candidates: Set[Candidate]) -> float:
        opponents = active_candidates.copy()
        opponents.remove(candidate)

        losses = [-self.delta(candidate, c2) for c2 in opponents]
        return max(losses)

    def minimax(self, active_candidates: Set[Candidate]) -> List[Candidate]:
        if len(active_candidates) == 1:
            return list(active_candidates)

        ac = active_candidates.copy()
        max_losses: List[Tuple[Candidate, float]] = [(ci, self.max_loss(ci, ac)) for ci in ac]

        max_losses.sort(key=lambda x: x[1])
        winner = max_losses[0][0]
        ac.remove(winner)
        return [winner] + self.minimax(ac)
