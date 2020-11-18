import numpy as np

def compute_variations(performance_measures: np.array) -> np.array:

    variations = []

    for i in range(1, len(performance_measures)):

        variations.append(performance_measures[i] - performance_measures[i-1])

    return np.array(variations)

def compute_gain(performance_variations: np.array) -> float:

    return np.sum(performance_variations)