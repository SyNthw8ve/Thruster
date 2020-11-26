from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from thruster.reaction_chamber.observer import Observer
from thruster.reaction_chamber.reaction import Reaction
from thruster.reaction_chamber.reactor import Reactor


class Chamber(py_environment.PyEnvironment):

    def __init__(self, reactor: Reactor, reaction: Reaction, observer: Observer):

        self._state = reactor
        self.observer = observer
        self.reaction = reaction

        self._observation_spec = self.observer.get_observation_spec()
        self._action_spec = self._state.get_action_specs()

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):

        self._state.reset()
        self.reaction.reset()

        self._episode_ended = False

        return ts.restart(self.observer.observe())

    def _step(self, action):

        if self._episode_ended:

            self.reset()

        self._state.apply_reaction(action)
        self.reaction.read_reactor_state(self._state)



        