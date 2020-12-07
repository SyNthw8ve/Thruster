from numpy.core.numeric import outer
from tf_agents.utils import nest_utils
from tf_agents.networks.network import Network
from tf_agents.utils.common import scale_to_spec
from tf_agents.networks.utils import BatchSquash
from tf_agents.networks.encoding_network import EncodingNetwork
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, tanh
from tensorflow.keras.initializers import RandomUniform, VarianceScaling
from tensorflow import nest, float32, float64


class ActorCritic(Network):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=(75, 40),
                 dropout_layer_params=None,
                 activation_fn=relu,
                 enable_last_layer_zero_initializer=False,
                 name="ActorCritic"):

        super(ActorCritic, self).__init__(
            input_tensor_spec=observation_spec, state_spec=(), name=name)
        self._action_spec = action_spec

        flat_action_spec = nest.flatten(action_spec)

        if len(flat_action_spec) > 1:
            raise ValueError(
                'Only a single action is supported by this network')

        self._single_action_spec = flat_action_spec[0]

        if self._single_action_spec.dtype not in [float32, float64]:
            raise ValueError(
                'Only float actions are supported by this network.')

        #kernel_initializer = VarianceScaling(scale=1. / 3., mode)

        self._encoder = EncodingNetwork(observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            #kernel_initializer=kernel_initializer,
            batch_squash=False)

        initializer = RandomUniform(minval=-0.003, maxval=0.003)

        self._action_projection_layer = Dense(
            flat_action_spec[0].shape.num_elements(),
            activation=tanh,
            kernel_initializer=initializer,
            name='action')

    def call(self, observations, step_type=(), network_state=()):
        
        outer_rank =  nest_utils.get_outer_rank(observations, self.input_tensor_spec)

        batch_squash = BatchSquash(outer_rank)
        observations = nest.map_structure(batch_squash.flatten, observations)

        state, network_state = self._encoder(
            observations, step_type=step_type, network_state=network_state)

        actions = self._action_projection_layer(state)
        actions = scale_to_spec(actions, self._single_action_spec)
        actions = batch_squash.unflatten(actions)

        return nest.pack_sequence_as(self._action_spec, [actions]), network_state