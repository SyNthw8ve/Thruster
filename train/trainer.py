from thruster.fuel_storage.fuel import Fuel
from thruster.reaction_chamber.chamber import Chamber
from thruster.reaction_chamber.observer import Observer
from thruster.reaction_chamber.propulsion import Propulsion
from thruster.reaction_chamber.reactor import Reactor


class Trainer:

    def __init__(self, fuel: Fuel, observer: Observer, propulsion: Propulsion, reactor: Reactor, 
                    network, network_params) -> None:

        self.train_chamber = Chamber(reactor, propulsion, observer, fuel)
        self.eval_chamber = Chamber(reactor, propulsion, observer, fuel)
