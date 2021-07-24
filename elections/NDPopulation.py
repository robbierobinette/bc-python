from .Population import Population
from .Voter import Voter, UnitVoter
from .Ideology import Ideology
from .Party import Party
import numpy as np
from .PopulationGroup import Independents


class NDPopulation(Population):
    def __init__(self, location: np.array, scale: np.array):
        super().__init__(Independents)
        self.location = location
        self.scale = scale
        self.dim = location.shape[0]

    def unit_sample_voter(self) -> UnitVoter:
        ideology = np.random.normal(loc=self.location, scale=self.scale)
        return UnitVoter(Ideology(ideology))
