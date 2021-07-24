from typing import List, Set
from .Candidate import Candidate
from .CandidateScore import CandidateScore
from .Voter import Voter
from .ElectionConfig import ElectionConfig


class Ballot:
    def __init__(self, voter: Voter, candidates: List[Candidate], config: ElectionConfig):
        self.voter = voter

        scores = list(map(lambda c: voter.score(c, config), candidates))
        cs = list(map(lambda c: CandidateScore(c[0], c[1]), zip(candidates, scores)))
        cs.sort(key=lambda c: c.score, reverse=True)
        self.ordered_candidates = cs

    def active_choice(self, active_candidates: Set[Candidate]) -> Candidate:
        for c in self.ordered_candidates:
            if c.candidate in active_candidates:
                return c.candidate
        assert(False, "no candidate in active candidates")

    def print(self):
        for cs in self.ordered_candidates:
            print("\t %6s ideology: % 7.2f score: % 7.2f" % (cs.candidate.name, cs.candidate.ideology.vec[0], cs.score))

