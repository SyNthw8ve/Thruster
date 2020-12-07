import numpy as np
from numpy.core.fromnumeric import var
import tensorflow as tf

from tf_agents.specs import array_spec

from thruster.reaction_chamber.observer import Observer
from thruster.reaction_chamber.reactor import Reactor


class GObserver(Observer):

    def __init__(self, reactor: Reactor) -> None:

        super().__init__(reactor)

    def observe(self):

        instances = np.array(self.observable.reactant.get_instances())

        if len(instances) < 2:

            mean_vector = np.zeros((2,), dtype='float32')
            var_vector = np.zeros((2,), dtype='float32')
            cov_matrix = np.zeros((2,2), dtype='float32')

        else:

            mean_vector = np.mean(instances, dtype='float32', axis=0)
            var_vector = np.var(instances, dtype='float32', axis=0)

            cov_matrix = np.cov(instances.T).astype('float32')

        #return np.append(mean_vector, var_vector)
        return cov_matrix

    def get_observation_spec(self) -> array_spec.ArraySpec:
        return array_spec.BoundedArraySpec(
            shape=(2,2), dtype='float32', name='observation')
