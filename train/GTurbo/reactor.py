from typing import Dict
import numpy as np

from thruster.reaction_chamber.reactor import Reactor
from tf_agents.specs import array_spec
from algorithms.gturbo.gturbo import GTurbo


class GReactor(Reactor):

    def __init__(self, initial_params, params_domain: Dict[str, np.array]) -> None:

        self.params_domain = params_domain

        super().__init__(GTurbo, initial_params)

    def run_initial_params(self, data):

        for instance in data:

            self.reactant.add_fuel(instance)

    def run(self, parameters, data):

        new_params = self.initial_params.copy()

        new_params['epsilon_b'] = parameters[0]
        new_params['lam'] = int(parameters[1])
        new_params['max_age'] = int(parameters[2])
        new_params['r0'] = parameters[3]

        self.reactant = GTurbo(**new_params)

        for instance in data:

            self.reactant.add_fuel(instance)

    def get_current_params(self) -> np.array:

        return np.array([self.reactant.epsilon_b, self.reactant.lam,
                         self.reactant.max_age, self.reactant.r0])

    def get_initial_params(self) -> np.array:

        return np.array([self.initial_params['epsilon_b'], self.initial_params['lam'],
                         self.initial_params['max_age'], self.initial_params['r0']])

    def get_action_specs(self) -> array_spec.BoundedArraySpec:

        return array_spec.BoundedArraySpec(
            shape=(4,), dtype='float32', minimum=self.params_domain['min'],
            maximum=self.params_domain['max'], name='action')
