from thruster.reaction_chamber.reactor import Reactor
from abc import ABC, abstractmethod
from tf_agents.specs import array_spec

class Observer(ABC):

    def __init__(self, reactor: Reactor) -> None:

        self.observable = reactor

    @abstractmethod
    def observe(self):
        pass

    @abstractmethod
    def get_observation_spec(self) -> array_spec.BoundedArraySpec:
        pass