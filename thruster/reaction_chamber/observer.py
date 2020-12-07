from thruster.reaction_chamber.reactor import Reactor
from thruster.fuel_storage.fuel import Fuel
from abc import ABC, abstractmethod
from tf_agents.specs import array_spec

class Observer(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def observe(self, current_params, fuel: Fuel, reward: float):
        pass

    @abstractmethod
    def get_observation_spec(self) -> array_spec.BoundedArraySpec:
        pass