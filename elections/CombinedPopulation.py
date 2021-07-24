from .Voter import *
from .PopulationGroup import *
from typing import List
from random import uniform
from .Population import Population


class CombinedPopulation(Population):
    def __init__(self, population_groups: List[PopulationGroup]):
        super().__init__(Independents)
        self.population_groups = population_groups
        self.combined_weight = sum([a.weight for a in population_groups])

        self.pg_dict = {}
        for g in population_groups:
            self.pg_dict[g.party] = g

        self.democrats = self.pg_dict[Democrats]
        self.republicans = self.pg_dict[Republicans]
        self.independents = self.pg_dict[Independents]

    def get_weighted_population(self) -> PopulationGroup:
        x = uniform(0, self.combined_weight)
        i = 0
        while x > 0:
            x -= self.population_groups[i].weight
            if x < 0:
                return self.population_groups[i]
            i += 1

        raise Exception("can't get weighted population")

    def sample_voter(self):
        p_group = self.get_weighted_population()
        return p_group.partisan_sample_voter()
