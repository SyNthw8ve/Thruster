import tensorflow as tf

from abc import ABC, abstractmethod

class Fuel(ABC):

    def __init__(self, file_name: str) -> None:

        self.data = self.load_data(file_name)
        self.iterator = iter(self.data)

    @abstractmethod
    def load_data(self, file_name: str):
        pass

    def get_fuel(self):

        try:

            return next(self.iterator)

        except StopIteration:
            
            return None

    