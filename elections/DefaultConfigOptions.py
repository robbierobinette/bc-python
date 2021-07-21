from PopulationGroup import *
from CombinedPopulation import *


unit_election_config = ElectionConfig(
    partisanship=0,
    stddev=1,
    skew_factor=0,
    primary_skew=0,
    party_loyalty=0,
    independent_bonus=0,
    wasted_vote_factor=0,
    uncertainty=.20
)

dw_nominate_election_config = ElectionConfig(
    partisanship=30,
    stddev=12,
    skew_factor=.5,
    primary_skew=12,
    party_loyalty=30,
    independent_bonus=20,
    wasted_vote_factor=10,
    uncertainty=15)

_cc = dw_nominate_election_config
_population_groups = [
    PopulationGroup(Republicans, _cc.partisanship, _cc.stddev, .4, _cc.primary_skew),
    PopulationGroup(Independents, 0, _cc.stddev, .4, 0),
    PopulationGroup(Democrats, -_cc.partisanship, _cc.stddev, .4, -_cc.primary_skew),
]

combined_population = CombinedPopulation(_population_groups)
