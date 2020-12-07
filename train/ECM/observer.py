import numpy as np
from numpy.core.fromnumeric import var
import tensorflow as tf

from tf_agents.specs import array_spec

from thruster.reaction_chamber.observer import Observer
from train.ECM.reactor import EReactor
from train.fuel2d import Fuel


class EObserver(Observer):

    def __init__(self) -> None:

        super().__init__()

    def observe(self, current_params, fuel: Fuel, reward: float):

        return {'static_state': fuel.get_statistics(), 'current_params': current_params,
                'current_score': reward}

    def get_observation_spec(self) -> array_spec.ArraySpec:
        return {
            'static_state': array_spec.ArraySpec((16,), np.float32),
            'current_params': array_spec.ArraySpec((1,), np.float32),
            'current_score': array_spec.ArraySpec((1,), np.float32)
        }
