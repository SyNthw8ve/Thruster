import numpy as np

from abc import ABC, abstractmethod
from thruster.reaction_chamber.reactor import Reactor

import numpy

class Reaction(ABC):

    def __init__(self) -> None:

        self.reactions: numpy.array = []

    @abstractmethod
    def get_reaction_value(self) -> float:
        pass

    @abstractmethod
    def get_reaction_reward(self) -> float:
        pass

    @abstractmethod
    def read_reactor_state(self, reactor: Reactor) -> None:
        pass

    def reset(self) -> None:

        self.reactions = []
