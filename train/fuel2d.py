import numpy as np

from thruster.fuel_storage.fuel import Fuel
from util.readers.reader_2d import Cluster2DReader
from sklearn.preprocessing import normalize
from scipy.stats import kurtosis, skew


class Fuel2D(Fuel):

    def __init__(self, folder, file, num_instances: int) -> None:

        super().__init__(folder, file, num_instances)

    def load_data(self):

        data = Cluster2DReader.read_data(self.file_name)

        return normalize(data)

    def get_partial_data_statistics(self, data):

        data_kurtosis = kurtosis(data)
        data_skew = skew(data)

        skew_min = np.min(data_skew)
        skew_max = np.max(data_skew)
        skew_mean = np.mean(data_skew)
        skew_std = np.std(data_skew)

        kurtosis_min = np.min(data_kurtosis)
        kurtosis_max = np.max(data_kurtosis)
        kurtosis_mean = np.mean(data_kurtosis)
        kurtosis_std = np.std(data_kurtosis)

        instances = len(data)
        log_instances = np.log(instances)

        data_dimension = 2
        log_dimension = np.log(2)

        inverse_data_dimension = 1 / data_dimension
        inverse_log_dimension = np.log(inverse_data_dimension)

        return np.array([instances, log_instances, data_dimension, log_dimension, 
            inverse_data_dimension, inverse_log_dimension, kurtosis_min, kurtosis_max, kurtosis_mean, kurtosis_std,
            skew_min, skew_max, skew_mean, skew_std])

    def get_full_data_statistics(self) -> np.array:

        return self.get_partial_data_statistics(self.data)
