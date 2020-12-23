import random as rd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tf_agents.trajectories import time_step as ts
from thruster.reaction_chamber.chamber import Chamber
from train.ECM.propulsion import EPropulsion
from train.ECM.reactor import EReactor
from train.ECM.observer import EObserver
from train.ECM.dynamic_observer import EDynamicObserver
from thruster.fuel_storage.fuel import Fuel
from thruster.fuel_storage.injector import Injector
from util.metrics.stability_analysis import compute_variations

from tests.policy_test import DynamicPolicyTest, PolicyTest


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

        initial_observation = self.observer.observe(
            initial_params, self.fuel, reward)

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


class ECMDynamicPolicyTest(DynamicPolicyTest):

    def __init__(self, reactor: EReactor, injector: Injector, propulsion: EPropulsion,
                 observer: EDynamicObserver, policy_path: str, num_batches: int) -> None:

        super().__init__(reactor, injector, propulsion, observer, policy_path)

        self.batches = []

        for _ in range(num_batches):

            self.injector.inject()
            self.batches.append(self.injector.current_data)

    def _build_time_step(self, step_type_val, reward_val, discount_val, observation_val):

        step_type = tf.convert_to_tensor(
            [step_type_val], dtype=tf.int32, name='step_type')

        reward = tf.convert_to_tensor(
            [reward_val], dtype=tf.float32, name='reward')

        discount = tf.convert_to_tensor(
            [discount_val], dtype=tf.float32, name='discount')

        observation = {
           
            'data_state': tf.convert_to_tensor([observation_val['data_state']], dtype=tf.float32, name="observtion/data_state"),
            'current_params': tf.convert_to_tensor([observation_val['current_params']], dtype=tf.float32, name="observtion/current_params"),
            'current_score': tf.convert_to_tensor([observation_val['current_score']], dtype=tf.float32, name="observtion/current_score"),
        }

        return ts.TimeStep(step_type, reward, discount, observation)

    def run_policy(self, grid_index: int):

        scores_before_policy = []
        scores_after_policy = []

        policy_state = self.policy.get_initial_state(1)
        current_params = grid_index
        step_type = 0

        for batch in self.batches:

            self.reactor.run(current_params, batch)
            reward = self.propulsion.get_propulsion_reward(self.reactor)

            scores_before_policy.append(reward)

            observation = self.observer.observe_batch(current_params=self.reactor.get_current_params(),
                data=batch, reward=reward, injector=self.injector)

            time_step = self._build_time_step(step_type_val=step_type, reward_val=reward, discount_val=1.0,
                observation_val=observation)

            action_step = self.policy.action(time_step, policy_state)

            current_params = action_step.action
            policy_state = action_step.state
            step_type = 1

            self.reactor.run(current_params, batch)
            reward = self.propulsion.get_propulsion_reward(self.reactor)

            scores_after_policy.append(reward)

        return scores_before_policy, scores_after_policy

    def run_no_policy(self, grid_index: int):
        
        scores = []
        
        for batch in self.batches:

            self.reactor.run(grid_index, batch)
            score = self.propulsion.get_propulsion_reward(self.reactor)

            scores.append(score)

        return scores

    def full_run_plot(self, grid_index, save_path_scores, save_path_var):

        print("Running No Policy...")
        scores_no_policy = self.run_no_policy(grid_index)

        print("Running Policy...")
        _, scores_after_policy = self.run_policy(grid_index)

        print("Plotting...")
        plt.plot(scores_no_policy, color='green', label='no policy')
        plt.plot(scores_after_policy, color='red', label='after policy')

        plt.savefig(save_path_scores)
        plt.close()

        scores_no_policy_var = compute_variations(scores_no_policy)
        scores_after_policy_var = compute_variations(scores_after_policy)

        plt.plot(scores_no_policy_var, color='orange', label='no policy')
        plt.plot(scores_after_policy_var, color='blue', label='after policy')

        plt.savefig(save_path_var)

        
