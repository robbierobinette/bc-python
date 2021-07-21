class ElectionConfig:
    def __init__(self,
                 partisanship: float,
                 stddev: float,
                 skew_factor: float,
                 primary_skew: float,
                 party_loyalty: float,
                 independent_bonus: float,
                 wasted_vote_factor: float,
                 uncertainty: float):
        self.partisanship = partisanship
        self.stddev = stddev
        self.skew_factor = skew_factor
        self.primary_skew = primary_skew
        self.party_loyalty = party_loyalty
        self.independent_bonus = independent_bonus
        self.wasted_vote_factor = wasted_vote_factor
        self.uncertainty = uncertainty


