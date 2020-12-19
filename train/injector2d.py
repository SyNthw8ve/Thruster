from os import set_blocking
import random as rd

from thruster.fuel_storage.fuel import Fuel
from thruster.fuel_storage.injector import Injector

class Injector2D(Injector):

    def __init__(self, min_quantity: int, max_quantity: int, fuel: Fuel) -> None:
        super().__init__(min_quantity, max_quantity, fuel)

    def inject(self):
        
        quantity = rd.randint(self.min_quantity, self.max_quantity)
        self.current_data = rd.sample(self.fuel.data, quantity)

    def get_statistics(self):
        
        return self.fuel.get_partial_data_statistics(self.current_data)