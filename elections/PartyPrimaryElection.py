from CombinedPopulation import Democrats, Republicans, CombinedPopulation
from Election import *
from Party import Party
from PluralityElection import PluralityElection
from PopulationGroup import PopulationGroup
from ElectionConfig import ElectionConfig


class PartyPrimaryElection(Election):
    def __init__(self, ballots: List[Ballot], candidates: Set[Candidate], combined_pop: CombinedPopulation,
                 config: ElectionConfig):
        super().__init__(ballots, candidates)
        self.primaries = {}
        self.combined_pop = combined_pop
        self.config = config

        dem_candidate = self.result_for_party(Democrats)
        rep_candidate = self.result_for_party(Republicans)
        self.general_candidates = {dem_candidate, rep_candidate}
        self.general = PluralityElection(ballots, self.general_candidates)

    def result(self) -> ElectionResult:
        return self.general.result()

    def result_for_party(self, party: Party) -> Candidate:
        pop = self.combined_pop.pg_dict[party]
        primary_pop = PopulationGroup(pop.party, pop.mean + pop.primary_shift, pop.stddev, pop.weight, 0)
        party_candidates = set(filter(lambda c: c.party == party, self.candidates))
        party_ballots = [Ballot(primary_pop.partisan_sample_voter(), list(party_candidates), self.config) for x in
                         range(len(self.ballots))]
        primary = PluralityElection(party_ballots, party_candidates)
        self.primaries[party] = primary
        return primary.result().winner()

    def print(self):
        for k in self.primaries.keys():
            print("Primary: %12s" % k.name)
            self.primaries[k].print()

        print("General:")
        self.general.print()
