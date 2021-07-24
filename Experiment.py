
from elections.Ideology import Ideology
import random
from typing import List

import numpy as np

from elections.Candidate import Candidate
from elections.Ideology import Ideology
from elections.NDPopulation import NDPopulation
from elections.PopulationGroup import Independents
from network.Tensor import Tensor
from ExperimentalConfig import ExperimentalConfig

class Experiment:
    def __init__(self, config: ExperimentalConfig):
        self.config = config