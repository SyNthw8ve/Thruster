from thruster.agents.agent_network import AgentNetwork
from thruster.networks.q_rnn import QRnn
from tf_agents.agents.dqn import dqn_agent


class DqnAgent(AgentNetwork):

    def __init__(self, q_network: QRnn, time_step_spec, action_spec, optimizer, td_errors_loss_fn, train_step_counter) -> None:

        self.agent = dqn_agent.DqnAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=q_network.q_rnn,
            optimizer=optimizer,
            td_errors_loss_fn=td_errors_loss_fn,
            train_step_counter=train_step_counter)

        self.agent.initialize()
