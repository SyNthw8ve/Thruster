from thruster.reaction_chamber.dynamic_observer import DynamicObserver
import numpy as np
import random as rd

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

from thruster.reaction_chamber.observer import Observer
from thruster.reaction_chamber.propulsion import Propulsion
from thruster.reaction_chamber.reactor import Reactor
from thruster.fuel_storage.injector import Injector

class DynamicChamber(py_environment.PyEnvironment):

    def __init__(self, reactor: Reactor, propulsion: Propulsion, injector: Injector, observer: DynamicObserver, episode_lenght: int = 200):

        self._state = np.array(list(rd.choice(reactor.param_grid).values()))

        self._data_state = None

        self.propulsion = propulsion
        self.reactor = reactor
        self.injector = injector
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
        self.injector.inject()

        self._state = np.array(
            list(rd.choice(self.reactor.param_grid).values()))

        self._data_state = self.injector.get_statistics()

        self._episode_ended = False
        self.episode_iteration = 0

        self.previous_action = -1

        initial_observation = self.observer.observe(
            current_params=self._state, injector=self.injector, reward=0)
        return ts.restart(initial_observation)

    def _step(self, action):

        if self._episode_ended:

            self.reset()

        self.reactor.run(action, self.injector.current_data)

        reward = self.propulsion.get_propulsion_reward(
            self.reactor)

        observation = self.observer.observe(
            current_params=self._state, injector=self.injector, reward=reward)

        self._state = self.reactor.get_current_params()

        if (self.episode_iteration < self.episode_lenght) and (self.previous_action != action):

            self.episode_iteration += 1
            self.previous_action = action
            return ts.transition(observation, reward=reward, discount=1.0)

        else:

            self._episode_ended = True
            return ts.termination(observation, reward=reward)
