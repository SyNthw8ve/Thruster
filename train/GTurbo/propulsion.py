from numpy.core.fromnumeric import var
from thruster.reaction_chamber.propulsion import Propulsion
from thruster.reaction_chamber.reactor import Reactor

from util.metrics.cluster_analysis import eval_cluster
from util.metrics.stability_analysis import compute_gain, compute_variations

class GPropulsion(Propulsion):

    def __init__(self) -> None:

        super().__init__()

    def get_propulsion_value(self) -> float:
        
        variations = compute_variations(self.propulsions)
        return compute_gain(variations)

    def get_propulsion_reward(self) -> float:
        pass

    def read_reactor_state(self, reactor: Reactor) -> None:
        
        score_t = eval_cluster(reactor.reactant)
        self.propulsions.append(score_t)