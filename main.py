import tensorflow as tf
import random as rd

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.utils import common

from thruster.agents.dqn_agent import DqnAgent
from thruster.networks.q_rnn import QRnn
from thruster.reaction_chamber.dynamic_chamber import DynamicChamber
from thruster.reaction_chamber.chamber import Chamber

from train.trainers.q_trainer import QTrainer
from train.ECM.propulsion import EPropulsion
from train.ECM.reactor import EReactor
from train.ECM.observer import EObserver
from train.ECM.dynamic_observer import EDynamicObserver
from train.fuel2d import Fuel2D
from train.injector2d import Injector2D

from util.params.params import build_tests

from tests.ecm_policy_test import ECMDynamicPolicyTest

data_folder = './data/2D/ASets'
file_name = 'a1.txt'

num_instances = 2000

param_grid = build_tests(
    {'distance_threshold': [1, 0.1, 0.001, 0.0001, 0.01, 0.00005, 0.2, 0.002, 1.5]})

fuel_train = Fuel2D(folder=data_folder, file=file_name,
                    num_instances=num_instances)
fuel_eval = Fuel2D(folder=data_folder, file=file_name,
                   num_instances=num_instances)

injector_train = Injector2D(
    fuel=fuel_train, min_quantity=10, max_quantity=num_instances)
injector_eval = Injector2D(
    fuel=fuel_eval, min_quantity=10, max_quantity=num_instances)

reactor_train = EReactor(param_grid=param_grid)
reactor_eval = EReactor(param_grid=param_grid)

propulsion_train = EPropulsion()
propulsion_eval = EPropulsion()

dynamic_observer = EDynamicObserver()

train_chamber = DynamicChamber(reactor=reactor_train, propulsion=propulsion_train, observer=dynamic_observer,
                               injector=injector_train, episode_lenght=200)
eval_chamber = DynamicChamber(reactor=reactor_eval, propulsion=propulsion_eval, observer=dynamic_observer,
                              injector=injector_eval, episode_lenght=200)

train_chamber_tf = TFPyEnvironment(train_chamber)
eval_chamber_tf = TFPyEnvironment(eval_chamber)

q_rnn_args = {
    'preprocessing_layers': {
        'data_state': tf.keras.layers.Flatten(),
        'current_params': tf.keras.layers.Flatten(),
        'current_score': tf.keras.layers.Flatten()
    },
    'preprocessing_combiner': tf.keras.layers.Concatenate(axis=-1),
    'lstm_size': (16, 32),
    'observation_spec': train_chamber_tf.observation_spec(),
    'action_spec': train_chamber_tf.action_spec()
}

q_net = QRnn(**q_rnn_args)

dqn_agent_args = {
    'q_network': q_net,
    'time_step_spec': train_chamber_tf.time_step_spec(),
    'action_spec': train_chamber_tf.action_spec(),
    'optimizer': tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
    'td_errors_loss_fn': common.element_wise_squared_loss,
    'train_step_counter': tf.Variable(0)
}

dqn_agent = DqnAgent(**dqn_agent_args)

trainer = QTrainer(train_chamber=train_chamber_tf,
                   eval_chamber=eval_chamber_tf, wrapper_agent=dqn_agent)

trainer.run(replay_buffer_max_length=100000, num_iterations=20000,
            log_interval=500, eval_interval=1000, num_eval_episodes=20,
            collect_steps_per_iteration=1, batch_size=32, 
            initial_collect_steps=1000, policy_save_path='./policies/q_rnn_dynamic_cov')

d_test = ECMDynamicPolicyTest(reactor=reactor_eval, injector=injector_eval,
                           propulsion=propulsion_eval, observer=dynamic_observer,
                           policy_path='./policies/q_rnn_dynamic_cov', num_batches=100)

d_test.full_run_plot(3, './results/plots/results_a3_cov.png', './results/plots/results_a3_cov_var.png')