from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

from thruster.reaction_chamber.observer import Observer
from thruster.reaction_chamber.propulsion import Propulsion
from thruster.reaction_chamber.reactor import Reactor
from thruster.fuel_storage.fuel import Fuel

class Chamber(py_environment.PyEnvironment):

    def __init__(self, reactor: Reactor, propulsion: Propulsion, observer: Observer, fuel: Fuel):

        self._state = reactor
        self.observer = observer
        self.propulsion = propulsion
        self.fuel = fuel

        self._observation_spec = self.observer.get_observation_spec()
        self._action_spec = self._state.get_action_specs()

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):

        self._state.reset()
        self.propulsion.reset()
        self.fuel.re_fuel()

        self._episode_ended = False

        return ts.restart(self.observer.observe())

    def _step(self, action):

        if self._episode_ended:

            self.reset()

        self._state.apply_reaction(action)
        self.propulsion.read_reactor_state(self._state)
        observation = self.observer.observe()

        next_instance = self.fuel.get_fuel()
        
        if next_instance == None:

            self._episode_ended = True

            reward = self.propulsion.get_reaction_value()
            return ts.termination(observation, reward=reward)

        else:

            self._state.reactant.add_fuel(next_instance)
            return ts.transition(observation, reward=0, discount=1.0)
            

       
        




        