import numpy as np
from thruster.reaction_chamber.propulsion import Propulsion
from thruster.reaction_chamber.reactor import Reactor

from util.metrics.cluster_analysis import eval_cluster_kmeans
from util.metrics.stability_analysis import compute_gain, compute_variations

class KPropulsion(Propulsion):

    def __init__(self) -> None:

        super().__init__()

    def get_propulsion_value(self) -> float:
        
        variations = compute_variations(self.propulsions)
        return np.sum(self.propulsions)
        #return compute_gain(variations)

    def get_propulsion_reward(self, reactor: Reactor, data) -> float:
        
        return eval_cluster_kmeans(reactor.reactant, data)

    def read_reactor_state(self, reactor: Reactor) -> None:
        
        score_t = eval_cluster_kmeans(reactor.reactant)
        self.propulsions = np.append(self.propulsions, [score_t])