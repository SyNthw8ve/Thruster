import os
import numpy as np
import tensorflow as tf

from train.GTurbo.fuel2d import GFuel2D
from train.GTurbo.observer import GObserver
from train.GTurbo.propulsion import GPropulsion
from train.GTurbo.reactor import GReactor

from train.KMeans.propulsion import KPropulsion
from train.KMeans.reactor import KReactor

from train.ECM.propulsion import EPropulsion
from train.ECM.reactor import EReactor

from thruster.reaction_chamber.chamber import Chamber
from thruster.networks.q_network import QNetwork

from util.readers.reader_2d import Cluster2dReader
from util.params.params import build_tests

data_folder = './data/2D/ASets'

file_name = 'a1.txt'

file = os.path.join(data_folder, file_name)

data = Cluster2dReader.read_data(file)

gFuel_train = GFuel2D(file_name=file, num_instances=200)
gFuel_eval = GFuel2D(file_name=file, num_instances=200)

""" reactor = GReactor(initial_params={'epsilon_b': 0.001, 'lam': 20, 'max_age': 20, 'r0': 0.01, 
                    'epsilon_n': 0.0, 'beta': 0.995, 'alpha': 0.95, 'dimensions': 2},
                   params_domain={'max': np.array([1.0, 500, 500, 1.0]), 'min': np.array([0.001, 20, 20, 0.001])})

observer = GObserver(reactor=reactor)
propulsion = GPropulsion() """

param_grid = build_tests({'distance_threshold': [1, 0.1, 0.001, 0.0001, 0.01, 0.00005, 0.2, 0.002, 1.5]})

reactor_train = EReactor(param_grid=param_grid)
reactor_eval = EReactor(param_grid=param_grid)
propulsion_train = EPropulsion()
propulsion_eval = EPropulsion()

train_chamber = Chamber(reactor=reactor_train, propulsion=propulsion_train,
                         fuel=gFuel_train, episode_lenght=500)
eval_chamber = Chamber(reactor=reactor_eval, propulsion=propulsion_eval,
                        fuel=gFuel_eval, episode_lenght=500)

actor_critic = QNetwork(train_chamber, eval_chamber)

actor_critic.train('ecm_2d_a1_static')
