import tensorflow as tf

from abc import ABC, abstractmethod

from thruster.fuel_storage.fuel import Fuel
from thruster.fuel_storage.injector import Injector
from thruster.reaction_chamber.reactor import Reactor
from thruster.reaction_chamber.propulsion import Propulsion
from thruster.reaction_chamber.observer import Observer
from thruster.reaction_chamber.dynamic_observer import DynamicObserver


class PolicyTest(ABC):

    def __init__(self, reactor: Reactor, fuel: Fuel, propulsion: Propulsion, observer: Observer,
                 policy_path: str) -> None:

        self.reactor = reactor
        self.fuel = fuel
        self.propulsion = propulsion
        self.observer = observer

        self.policy = tf.compat.v2.saved_model.load(policy_path)

    @abstractmethod
    def run_policy(self):
        pass

    @abstractmethod
    def run_no_policy(self):
        pass


class DynamicPolicyTest(ABC):

    def __init__(self, reactor: Reactor, injector: Injector, propulsion: Propulsion, observer: DynamicObserver,
                 policy_path: str) -> None:

        self.reactor = reactor
        self.injector = injector
        self.propulsion = propulsion
        self.observer = observer

        self.policy = tf.compat.v2.saved_model.load(policy_path)

    @abstractmethod
    def run_policy(self):
        pass

    @abstractmethod
    def run_no_policy(self):
        pass
