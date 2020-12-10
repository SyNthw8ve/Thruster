from train.trainer import Trainer

from thruster.reaction_chamber.chamber import Chamber
from thruster.agents.agent_network import AgentNetwork

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common

class QTrainer(Trainer):

    def __init__(self, train_chamber: Chamber, eval_chamber: Chamber,
                 wrapper_agent: AgentNetwork) -> None:

        super().__init__(train_chamber, eval_chamber, wrapper_agent)

    def run(self, replay_buffer_max_length, num_iterations, 
                batch_size, log_interval, eval_interval, 
                num_eval_episodes, collect_steps_per_iteration, 
                initial_collect_steps, policy_save_path):

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.tf_agent.collect_data_spec,
            batch_size=self.train_chamber.batch_size,
            max_length=replay_buffer_max_length)

        random_policy = random_tf_policy.RandomTFPolicy(self.train_chamber.time_step_spec(),
                                                        self.train_chamber.action_spec(),)

        self._collect_data(self.train_chamber, random_policy,
                           replay_buffer, initial_collect_steps)

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)

        iterator = iter(dataset)

        self.tf_agent.train = common.function(self.tf_agent.train)

        self.tf_agent.train_step_counter.assign(0)

        avg_return = self._compute_avg_return(
            self.eval_chamber, self.tf_agent.policy, num_eval_episodes)
        returns = [avg_return]

        for _ in range(num_iterations):

            self._collect_data(self.train_chamber, self.tf_agent.collect_policy,
                               replay_buffer, collect_steps_per_iteration)

            experience, unused_info = next(iterator)
            train_loss = self.tf_agent.train(experience).loss

            step = self.tf_agent.train_step_counter.numpy()

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % eval_interval == 0:
                avg_return = self._compute_avg_return(
                    self.eval_chamber, self.tf_agent.policy, num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(
                    step, avg_return))
                returns.append(avg_return)

        self.wrapper_agent.save_policy(policy_save_path)