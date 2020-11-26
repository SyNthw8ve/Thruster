from thruster.fuel_storage.fuel import Fuel
from util.readers.setup_reader import DataInitializer

class GFuel(Fuel):

    def __init__(self, file_name: str) -> None:

        super().__init__(file_name)

    def load_data(self, file_name: str):
        
        return DataInitializer.read_openings(file_name)
