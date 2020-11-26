from abc import ABC, abstractmethod

class Reactant(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def apply_changes(self, param_change):
        pass

    @abstractmethod
    def add_fuel(self, fuel_data):
        pass

    @abstractmethod
    def get_instances(self):
        pass