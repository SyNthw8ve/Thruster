import numpy as np

from tf_agents.specs import array_spec

from thruster.reaction_chamber.dynamic_observer import DynamicObserver
from thruster.fuel_storage.injector import Injector

class EDynamicObserver(DynamicObserver):

    def __init__(self) -> None:

        super().__init__()

    def observe(self, current_params, injector: Injector, reward: float):

        return {'data_state': injector.get_statistics().astype('float32'), 'current_params': current_params.astype('float32'),
                'current_score': np.array(reward, dtype=np.float32)}

    def get_observation_spec(self):
        return {
            'data_state': array_spec.ArraySpec((14,), np.float32),
            'current_params': array_spec.ArraySpec((1,), np.float32),
            'current_score': array_spec.ArraySpec((), np.float32)
        }
