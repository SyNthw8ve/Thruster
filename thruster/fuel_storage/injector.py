from abc import ABC, abstractmethod
from thruster.fuel_storage.fuel import Fuel

class Injector(ABC):

    def __init__(self, min_quantity: int, max_quantity: int, fuel: Fuel) -> None:
        
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.fuel = fuel
        self.current_data = None

    @abstractmethod
    def inject(self):
        pass

    @abstractmethod
    def get_statistics(self):
        pass

    @abstractmethod
    def get_batch_statistics(self, data):
        pass