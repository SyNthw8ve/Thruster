import numpy as np
import random as rd

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec

from thruster.reaction_chamber.observer import Observer
from thruster.reaction_chamber.propulsion import Propulsion
from thruster.reaction_chamber.reactor import Reactor
from thruster.fuel_storage.fuel import Fuel

class Chamber(py_environment.PyEnvironment):

    def __init__(self, reactor: Reactor, propulsion: Propulsion, fuel: Fuel, observer: Observer, episode_lenght: int = 200):

        self._state = np.array(list(rd.choice(reactor.param_grid).values()))

        self._static_state = fuel.get_full_data_statistics()

        self.propulsion = propulsion
        self.reactor = reactor
        self.fuel = fuel
        self.observer = observer

        self.episode_lenght = episode_lenght
        self.episode_iteration = 0
        self.previous_action = -1

        self._observation_spec = self.observer.get_observation_spec()
        self._action_spec = self.reactor.get_action_spec()

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):

        self.propulsion.reset()
        self.fuel.re_fuel()

        self._state = np.array(
            list(rd.choice(self.reactor.param_grid).values()))

        self._episode_ended = False
        self.episode_iteration = 0
        self.previous_action = -1

        initial_observation = self.observer.observe(
            current_params=self._state, fuel=self.fuel, reward=0)
        return ts.restart(initial_observation)

    def _step(self, action):

        if self._episode_ended:

            self.reset()

        self.reactor.run(action, self.fuel.data)

        reward = self.propulsion.get_propulsion_reward(
            self.reactor)

        observation = self.observer.observe(
            current_params=self._state, fuel=self.fuel, reward=reward)

        self._state = self.reactor.get_current_params()

        if (self.episode_iteration < self.episode_lenght) and (self.previous_action != action):

            self.episode_iteration += 1
            self.previous_action = action
            return ts.transition(observation, reward=reward, discount=1.0)

        else:

            self._episode_ended = True
            return ts.termination(observation, reward=reward)
