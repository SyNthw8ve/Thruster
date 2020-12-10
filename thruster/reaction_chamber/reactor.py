import numpy as np

from abc import ABC, abstractmethod
from tf_agents.specs import array_spec
from thruster.reaction_chamber.reactant import Reactant

class Reactor(ABC):

    def __init__(self, reactor_class, param_grid) -> None:

        self.reactor_class = reactor_class
        self.param_grid = param_grid

        self.reactant = None

    def reset(self):
        
        self.reactant = None

    @abstractmethod
    def run(self, parameters, data):
        pass

    @abstractmethod
    def get_action_spec(self) -> array_spec.BoundedArraySpec:
        pass

    @abstractmethod
    def get_current_params(self) -> np.array:
        pass

    @abstractmethod
    def get_initial_params(self) -> np.array:
        pass