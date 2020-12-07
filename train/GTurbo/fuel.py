from thruster.fuel_storage.fuel import Fuel
from util.readers.setup_reader import DataInitializer

class GFuel(Fuel):

    def __init__(self, file_name: str, num_instances: int) -> None:

        super().__init__(file_name, num_instances)

    def load_data(self):
        
        return DataInitializer.read_openings(self.file_name)
