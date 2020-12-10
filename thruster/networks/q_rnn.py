from tf_agents.networks.q_rnn_network import QRnnNetwork

class QRnn:

    def __init__(self, preprocessing_layers, preprocessing_combiner, 
                    lstm_size, observation_spec, action_spec) -> None:
        
        self.q_rnn = QRnnNetwork(preprocessing_layers=preprocessing_layers,
                        preprocessing_combiner=preprocessing_combiner, 
                        lstm_size=lstm_size, input_tensor_spec=observation_spec,
                        action_spec=action_spec)

        