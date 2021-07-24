from .Party import Party, Independents
from .Ideology import Ideology
from .Candidate import Candidate
from .ElectionConfig import ElectionConfig
import random


class Voter:
    def __init__(self, ideology: Ideology):
        self.ideology = ideology

    def score(self, candidate: Candidate, config: ElectionConfig):
        pass


class PartisanVoter(Voter):
    def __init__(self, ideology: Ideology, party: Party):
        super(PartisanVoter, self).__init__(ideology)
        self.party = party

    def score(self, candidate: Candidate, config: ElectionConfig) -> float:
        score = -self.ideology.distance(candidate.ideology)
        score += candidate.quality

        if self.party == candidate.party:
            score += config.party_loyalty
        elif candidate.party == Independents:
            score += config.independent_bonus

        score += random.normalvariate(0, config.uncertainty)

        if candidate.party == Independents:
            score -= config.wasted_vote_factor

        return score


# just like a voter but without
class UnitVoter(Voter):
    def __init__(self, ideology: Ideology):
        super(UnitVoter, self).__init__(ideology)

    def score(self, candidate: Candidate, config: ElectionConfig) -> float:
        score = -self.ideology.distance(candidate.ideology)
        score += candidate.quality
        score += random.normalvariate(0, config.uncertainty)
        return score
