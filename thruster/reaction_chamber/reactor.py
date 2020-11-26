import numpy as np

from abc import ABC, abstractmethod
from tf_agents.specs import array_spec
from thruster.reaction_chamber.reactant import Reactant

class Reactor(ABC):

    def __init__(self, reactor_class, initial_params) -> None:

        self.reactor_class = reactor_class
        self.initial_params = initial_params

        self.reactant: Reactant = self.reactor_class(**self.initial_params)

    def reset(self):
        
        self.reactant = self.reactor_class(**self.initial_params)

    @abstractmethod
    def apply_reaction(self, action):
        pass

    @abstractmethod
    def get_action_specs(self) -> array_spec.BoundedArraySpec:
        pass

    @abstractmethod
    def get_current_params(self) -> np.array:
        pass