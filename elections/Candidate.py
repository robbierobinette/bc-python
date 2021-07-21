from Party import Party
from Ideology import Ideology


class Candidate:
    def __init__(self, name: str, party: Party, ideology: Ideology, quality: float):
        self.ideology = ideology
        self.name = name
        self.party = party
        self.quality = quality
