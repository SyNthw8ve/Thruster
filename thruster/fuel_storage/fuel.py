import os

from abc import ABC, abstractmethod

class Fuel(ABC):

    def __init__(self, folder: str, file: str, num_instances: int) -> None:

        self.file_name = os.path.join(folder, file)
        self.data = self.load_data()[:num_instances]
        self.iterator = iter(self.data)

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def get_statistics(self):
        pass

    def re_fuel(self):

        self.iterator = iter(self.data)

    def get_fuel(self):

        return next(self.iterator)


    