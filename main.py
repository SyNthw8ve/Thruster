import os
import numpy as np
import tensorflow as tf

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.agents.reinforce import reinforce_agent

from train.ECM.propulsion import EPropulsion
from train.ECM.reactor import EReactor
from train.ECM.observer import EObserver
from train.fuel2d import Fuel2D

from thruster.reaction_chamber.chamber import Chamber
from thruster.networks.actor_critic import ActorCritic

from util.readers.reader_2d import Cluster2dReader
from util.params.params import build_tests

data_folder = './data/2D/ASets'

file_name = 'a1.txt'

file = os.path.join(data_folder, file_name)

data = Cluster2dReader.read_data(file)

""" gFuel_train = GFuel2D(file_name=file, num_instances=200)
gFuel_eval = GFuel2D(file_name=file, num_instances=200)

reactor = GReactor(initial_params={'epsilon_b': 0.001, 'lam': 20, 'max_age': 20, 'r0': 0.01, 
                    'epsilon_n': 0.0, 'beta': 0.995, 'alpha': 0.95, 'dimensions': 2},
                   params_domain={'max': np.array([1.0, 500, 500, 1.0]), 'min': np.array([0.001, 20, 20, 0.001])})

observer = GObserver(reactor=reactor)
propulsion = GPropulsion()
"""
param_grid = build_tests(
    {'distance_threshold': [1, 0.1, 0.001, 0.0001, 0.01, 0.00005, 0.2, 0.002, 1.5]})

gFuel_train = Fuel2D(file_name=file, num_instances=20)
gFuel_eval = Fuel2D(file_name=file, num_instances=20)

reactor_train = EReactor(param_grid=param_grid)
reactor_eval = EReactor(param_grid=param_grid)

propulsion_train = EPropulsion()
propulsion_eval = EPropulsion()

observer = EObserver()

train_chamber = Chamber(reactor=reactor_train, propulsion=propulsion_train, observer=observer,
                        fuel=gFuel_train, episode_lenght=500)
eval_chamber = Chamber(reactor=reactor_eval, propulsion=propulsion_eval, observer=observer,
                       fuel=gFuel_eval, episode_lenght=500)

train_chamber_tf = TFPyEnvironment(train_chamber)
eval_chamber_tf = TFPyEnvironment(eval_chamber)

preprocessing_combiners = tf.keras.layers.Concatenate(axis=-1)

actor_net = ActorCritic(train_chamber_tf.observation_spec(
    ), train_chamber_tf.action_spec(), preprocessing_combiner=preprocessing_combiners)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

train_step_counter = tf.Variable(0.)

tf_agent = reinforce_agent.ReinforceAgent(
    train_chamber_tf.time_step_spec(),
    train_chamber_tf.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()
