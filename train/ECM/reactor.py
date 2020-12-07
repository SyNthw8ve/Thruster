from typing import Dict, List
import numpy as np

from thruster.reaction_chamber.reactor import Reactor
from tf_agents.specs import array_spec
from algorithms.ecm.ecm import ECM

class EReactor(Reactor):

    def __init__(self, param_grid: List) -> None:

        super().__init__(ECM, param_grid)

    def run(self, action, data):

        new_params = self.param_grid[action]

        self.reactant = ECM(**new_params)

        for instance in data:

            self.reactant.add_fuel(instance)

    def get_current_params(self) -> np.array:

        return np.array([self.reactant.distance_threshold])

    def get_initial_params(self) -> np.array:

        return np.array([self.initial_params['distance_threshold']])

    def get_action_specs(self) -> array_spec.BoundedArraySpec:

        return array_spec.BoundedArraySpec(
            shape=(1,), dtype='float32', minimum=0,
            maximum=len(self.param_grid) - 1, name='action')
