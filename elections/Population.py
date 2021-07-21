from Voter import Voter
from Party import Party
from typing import List


class Population:
    def __init__(self, party: Party):
        super().__init__()
        self.party = party

    def partisan_sample_voter(self) -> Voter:
        pass

    def unit_sample_voter(self) -> Voter:
        pass

    def generate_partisan_voters(self, n: int) -> List[Voter]:
        return list(map(lambda i: self.partisan_sample_voter(), range(n)))

    def generate_unit_voters(self, n: int) -> List[Voter]:
        return list(map(lambda i: self.unit_sample_voter(), range(n)))
