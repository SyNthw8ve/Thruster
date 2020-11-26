import numpy as np

from abc import ABC, abstractmethod
from thruster.reaction_chamber.reactor import Reactor

import numpy

class Propulsion(ABC):

    def __init__(self) -> None:

        self.propulsions: numpy.array = np.array([])

    @abstractmethod
    def get_propulsion_value(self) -> float:
        pass

    @abstractmethod
    def get_propulsion_reward(self) -> float:
        pass

    @abstractmethod
    def read_reactor_state(self, reactor: Reactor) -> None:
        pass

    def reset(self) -> None:

        self.propulsions = np.array([])
