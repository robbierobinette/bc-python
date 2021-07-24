from .Party import *
from .Voter import *
import random
import numpy as np


class PopulationGroup:
    def __init__(self, party: Party, mean: float, stddev: float, weight: float, primary_shift: float):
        self.party = party
        self.mean = mean
        self.stddev = stddev
        self.weight = weight
        self.primary_shift = primary_shift

    def partisan_sample_voter(self) -> PartisanVoter:
        i = Ideology(np.random.normal(loc=[self.mean], scale=[self.stddev]))
        return PartisanVoter(i, self.party)

    def unit_voter(self) -> UnitVoter:
        i = Ideology(np.random.normal(loc=[self.mean], scale=[self.stddev]))
        return UnitVoter(i)



