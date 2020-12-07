from abc import ABC, abstractmethod
from thruster.reaction_chamber.chamber import Chamber
from tf_agents.environments.tf_py_environment import TFPyEnvironment

class AgentNetwork:

    def __init__(self, train_chamber: Chamber, eval_chamber: Chamber, network, network_params) -> None:

        self.train_chamber = TFPyEnvironment(train_chamber)
        self.eval_chamber = TFPyEnvironment(eval_chamber)

        

    def train(self):

        pass

    def save_policy(self) -> None:

        pass
