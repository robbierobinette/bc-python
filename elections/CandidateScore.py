from .Candidate import Candidate


class CandidateScore:
    def __init__(self, candidate: Candidate, score: float):
        self.candidate = candidate
        self.score = score
