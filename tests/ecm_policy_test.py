import random as rd
import tensorflow as tf
from thruster.reaction_chamber.chamber import Chamber
import numpy as np

from tf_agents.trajectories import time_step as ts

from train.ECM.propulsion import EPropulsion
from train.ECM.reactor import EReactor
from train.ECM.observer import EObserver
from thruster.fuel_storage.fuel import Fuel

from tests.policy_test import PolicyTest


class ECMStaticPolicyTest(PolicyTest):

    def __init__(self, reactor: EReactor, fuel: Fuel, propulsion: EPropulsion, observer: EObserver,
                 policy_path: str) -> None:
        super().__init__(reactor, fuel, propulsion, observer, policy_path)

        self.data = fuel.data

    def run_policy(self, grid_index: int):

        self.reactor.run(grid_index, self.data)
        reward = self.propulsion.get_propulsion_reward(self.reactor)

        initial_params = self.reactor.get_current_params()
        print(initial_params, reward)

        initial_state = self.policy.get_initial_state(1)

        initial_observation = self.observer.observe(initial_params, self.fuel, reward)
    
        step_type = tf.convert_to_tensor(
            [0], dtype=tf.int32, name='step_type')

        reward = tf.convert_to_tensor(
            [reward], dtype=tf.float32, name='reward')

        discount = tf.convert_to_tensor(
            [1], dtype=tf.float32, name='discount')

        observation = {

            'static_state': tf.convert_to_tensor([initial_observation['static_state']], dtype=tf.float32, name="observtion/static_state"),
            'current_params': tf.convert_to_tensor([initial_observation['current_params']], dtype=tf.float32, name="observtion/current_params"),
            'current_score': tf.convert_to_tensor([initial_observation['current_score']], dtype=tf.float32, name="observtion/current_score"),
        }

        time_step = ts.TimeStep(step_type, reward, discount, observation)

        action_step = self.policy.action(time_step, initial_state)
        
        print(action_step.action)

        self.reactor.run(action_step.action, self.data)
        reward = self.propulsion.get_propulsion_reward(self.reactor)

        print(self.reactor.get_current_params(), reward)

    def run_no_policy(self, grid_index: int):

        self.reactor.run(grid_index, self.data)

        scores = self.propulsion.get_propulsion_reward(self.reactor)

        print(scores)   
