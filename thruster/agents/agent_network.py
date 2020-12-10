from abc import ABC, abstractmethod
from thruster.reaction_chamber.chamber import Chamber
from tf_agents.environments.tf_py_environment import TFPyEnvironment

class AgentNetwork(ABC):

    def __init__(self) -> None:
        pass
        

    @abstractmethod
    def save_policy(self, path: str) -> None:
        pass
