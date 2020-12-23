from thruster.fuel_storage.injector import Injector

from abc import ABC, abstractmethod

from tf_agents.specs import array_spec

class DynamicObserver(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def observe(self, current_params, injector: Injector, reward: float):
        pass

    @abstractmethod
    def observe_batch(self, current_params, injector: Injector, reward: float, data):
        pass

    @abstractmethod
    def get_observation_spec(self) -> array_spec.BoundedArraySpec:
        pass