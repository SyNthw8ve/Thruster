import numpy as np

from tf_agents.specs import array_spec

from thruster.reaction_chamber.observer import Observer
from thruster.reaction_chamber.reactor import Reactor

class GObserver(Observer):

    def __init__(self, reactor: Reactor) -> None:

        super().__init__(reactor)

    def observe(self):

        instances = np.array(self.observable.reactant.get_instances())
        return np.cov(instances.T)
        
    def get_observation_spec(self) -> array_spec.ArraySpec:
        array_spec.BoundedArraySpec(
            shape=(1024,1024), dtype='float32', name='observation')