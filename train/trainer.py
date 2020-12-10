from abc import ABC, abstractmethod

from thruster.reaction_chamber.chamber import Chamber
from thruster.agents.agent_network import AgentNetwork

from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_episode_driver, dynamic_step_driver

class Trainer(ABC):

    def __init__(self, train_chamber: Chamber, eval_chamber: Chamber,
                 wrapper_agent: AgentNetwork) -> None:

        self.train_chamber = train_chamber
        self.eval_chamber = eval_chamber

        self.tf_agent = wrapper_agent.agent
        self.wrapper_agent = wrapper_agent

    def _compute_avg_return(self, environment, policy, n_episodes):

        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()
        average_return = tf_metrics.AverageReturnMetric()

        observers = [num_episodes, env_steps, average_return]

        _driver = dynamic_episode_driver.DynamicEpisodeDriver(
            environment, policy, observers, num_episodes=n_episodes)

        final_time_step, _ = _driver.run()

        print('eval episodes = {0}: Average Return = {1}'.format(
            num_episodes.result().numpy(), average_return.result().numpy()))
        return average_return.result().numpy()

    def _collect_data(self, env, policy, buffer, steps):

        observers = [buffer.add_batch]

        driver = dynamic_step_driver.DynamicStepDriver(
            env, policy, observers, num_steps=steps)

        final_time_step, policy_state = driver.run()

    @abstractmethod
    def run(self, replay_buffer_max_length, num_iterations, batch_size, 
            log_interval, eval_interval, num_eval_episodes, collect_steps_per_iteration,
            policy_save_path):
        pass