from typing import Set, Callable

from .Ballot import Ballot
from .DefaultConfigOptions import *
from .Election import Election
from .ElectionResult import ElectionResult
from .HeadToHeadElection import HeadToHeadElection
from .InstantRunoffElection import InstantRunoffElection
from .PluralityElection import PluralityElection

class ElectionConstructor:
    def __init__(self, constructor: Callable[[List[Ballot], Set[Candidate]], Election], name: str):
        self.constructor = constructor
        self.name = name

    def run(self, ballots: List[Ballot], candidates: Set[Candidate]) -> ElectionResult:
        e = self.constructor(ballots, candidates)
        return e.result()


def construct_irv(ballots: List[Ballot], candidates: Set[Candidate]) -> Election:
    return InstantRunoffElection(ballots, candidates)


def construct_h2h(ballots: List[Ballot], candidates: Set[Candidate]) -> Election:
    return HeadToHeadElection(ballots, candidates)


def construct_plurality(ballots: List[Ballot], candidates: Set[Candidate]) -> Election:
    return PluralityElection(ballots, candidates)

