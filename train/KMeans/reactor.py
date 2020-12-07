from typing import Dict
import numpy as np

from thruster.reaction_chamber.reactor import Reactor
from tf_agents.specs import array_spec
from algorithms.gturbo.gturbo import GTurbo

from sklearn.cluster import KMeans


class KReactor(Reactor):

    def __init__(self, initial_params, params_domain: Dict[str, np.array]) -> None:

        self.params_domain = params_domain

        super().__init__(KMeans, initial_params)

    def run_initial_params(self, data):

        self.reactant.fit(data)

    def run(self, parameters, data):

        new_params = self.initial_params.copy()

        new_params['n_clusters'] = int(parameters[0])

        self.reactant = KMeans(**new_params)

        self.reactant.fit(data)

    def get_current_params(self) -> np.array:

        return np.array([self.reactant.n_clusters])

    def get_initial_params(self) -> np.array:

        return np.array([self.initial_params['n_clusters']])

    def get_action_specs(self) -> array_spec.BoundedArraySpec:

        return array_spec.BoundedArraySpec(
            shape=(1,), dtype='float32', minimum=self.params_domain['min'],
            maximum=self.params_domain['max'], name='action')
