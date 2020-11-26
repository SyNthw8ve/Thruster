from abc import ABC, abstractmethod

from tf_agents.specs import array_spec

class Reactor(ABC):

    def __init__(self, reactor_class, initial_params) -> None:

        self.reactor_class = reactor_class
        self.initial_params = initial_params

        self.reactant = self.reactor_class(**self.initial_params)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def apply_reaction(self, action):
        pass

    @abstractmethod
    def get_action_specs(self) -> array_spec.BoundedArraySpec:
        pass
