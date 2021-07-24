import matplotlib.pyplot as plt

from .Ballot import Ballot
from .DefaultConfigOptions import *
from .PartyPrimaryElection import PartyPrimaryElection


def main():
    ideology = []
    for i in range(1000):
        print(".")
        if (i + 1) % 100 == 0:
            print("")

        ideology.append(run_election())

    plt.hist([ideology],
             stacked=True,
             density=True,
             bins=30,
             color=["blue"],
             label=["representatives"],
             )
    plt.xlabel('ideology')
    plt.ylabel('count')
    plt.show()


def gen_candidates(population: PopulationGroup) -> List[Candidate]:
    cc = []
    for i in range(0, 3):
        v = population.partisan_sample_voter()
        cc.append(Candidate("%s-%d" % (population.party.short_name, i + 1), population.party, v.ideology, 0))
    return cc


def run_election() -> float:
    pop = combined_population
    voters = pop.generate_voters(1000)
    candidates = gen_candidates(pop.republicans) + gen_candidates(pop.democrats)
    ballots = list(map(lambda v: Ballot(v, candidates, default_election_config), voters))
    election = PartyPrimaryElection(ballots, set(candidates), pop, default_election_config)
    return election.result().winner().ideology.vec[0]


if __name__ == '__main__':
    main()
