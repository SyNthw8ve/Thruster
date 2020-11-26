from typing import Dict
import numpy as np

from thruster.reaction_chamber.reactor import Reactor
from tf_agents.specs import array_spec
from algorithms.gturbo.gturbo import GTurbo

class GReactor(Reactor):

    def __init__(self, initial_params, params_value_delta: Dict[str, np.array], params_limits: np.array) -> None:

        self.params_value_delta = params_value_delta
        self.params_limits = params_limits

        super().__init__(GTurbo, initial_params)

    def apply_reaction(self, action):

        current_params = self.get_current_params()
        action_vector = np.array(
            [action[0], int(round(action[1])), int(round(action[2])), action[3]])

        applied_change = current_params + action_vector

        for i in range(len(current_params)):

            if applied_change[i] <= self.params_limits[i]:

                applied_change[i] = current_params[i]

        self.reactant = self.reactant.apply_changes(applied_change)
        
    def current_params(self) -> np.array:

        return np.array([self.reactant.epsilon_b, self.reactant.lam,
                         self.reactant.max_age, self.reactant.r0])

    def get_action_specs(self) -> array_spec.BoundedArraySpec:

        return array_spec.BoundedArraySpec(
            shape=(len(self.reactant.initial_params),), dtype='float32', minimum=self.params_value_delta['min'],
            maximum=self.params_value_delta['max'], name='action')
