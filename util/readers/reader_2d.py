import numpy as np

class Cluster2DReader:

    @staticmethod
    def read_data(file_name: str) -> np.array:

        numerical_coordinates = []

        with open(file_name, 'r') as f:

            coordinates = f.readlines()

            parsed_coordinates = [coordinate.strip().split('   ')
                                  for coordinate in coordinates]
            numerical_coordinates = [np.array([int(coordinate[0]), int(
                coordinate[1])]) for coordinate in parsed_coordinates]

        return np.array(numerical_coordinates)


