import csv
from typing import List
from .CombinedPopulation import CombinedPopulation
from .PopulationGroup import PopulationGroup, Democrats, Republicans, Independents


class DistrictVotingRecord:
    def __init__(self,
                 district: str,
                 incumbent: str,
                 expected_lean: float,
                 d1: float, r1: float,
                 d2: float, r2: float):
        self.district = district
        self.incumbent = incumbent
        self.expected_lean = expected_lean
        self.d1 = d1
        self.r1 = r1
        self.d2 = d2
        self.r2 = r2

        l1 = .5 - d1 / (d1 + r1)
        l2 = .5 - d2 / (d2 + r2)
        self.lean = 100 * (l1 + l2) / 2

    def print(self) -> None:
        print("%6s %25s % 5.2f" % (self.district, self.incumbent, self.lean))

    def population(self, partisanship: float, skew_factor: float, stddev: float) -> CombinedPopulation:
        s = self
        r_pct = (s.r1 + s.r2) / 2 / 100
        d_pct = (s.d1 + s.d2) / 2 / 100

        i_weight = .20

        r_weight = max(0.05, (1 - i_weight) * r_pct)
        d_weight = max(0.05, (1 - i_weight) * d_pct)

        skew = (r_weight - d_weight) / 2.0 * skew_factor * 100

        rep = PopulationGroup(Republicans, partisanship + skew, stddev, r_weight, 12)
        dem = PopulationGroup(Democrats, -partisanship + skew, stddev, d_weight, -12)
        ind = PopulationGroup(Independents, 0 + skew, stddev, i_weight, 0)
        return CombinedPopulation([rep, dem, ind])


class DistrictData:
    def __init__(self, path: str):
        self.path = path
        self.dvr = {}
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if row[0] != 'district':
                    dvr = self.parse_row(row)
                    self.dvr[dvr.district] = dvr

    def parse_row(self, row: List[str]) -> DistrictVotingRecord:
        if row[2] == 'EVEN':
            lean = 0
        elif row[2][0] == 'R':
            lean = float(row[2][2:])
        else:
            lean = -float(row[2][2:])
        d1 = float(row[3])
        r1 = float(row[4])
        if row[5] == 'null':
            d2 = d1
            r2 = r1
        else:
            d2 = float(row[5])
            r2 = float(row[6])

        return DistrictVotingRecord(row[0], row[1], lean, d1, r1, d2, r2)


def main():
    dd = DistrictData("data-5vPn3.csv")
    print("got dd")
    for k, v in dd.dvr.items():
        v.print()


if __name__ == "__main__":
    main()
