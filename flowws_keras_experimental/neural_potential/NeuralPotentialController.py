from collections import namedtuple

import flowws
from flowws import Argument as Arg
import tensorflow as tf

from .NeuralPotentialDropout import LearnedDropout

CurrentStatus = namedtuple('CurrentStatus', ['mean_probability', 'error'])
LogPoint = namedtuple('LogPoint', ['mean_probability', 'controller_output'])

class NeuralPotentialControlCallback(tf.keras.callbacks.Callback):
    def __init__(self, k_p, tau, setpoint=.5, period=1, log=False):
        self.k_p = k_p
        self.tau = tau
        self.setpoint = setpoint
        self.period = period
        self.log = log
        self.batch_count = None

        self.controlled_layers = []
        self.initial_value = None
        self.integrated_error = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.log_history = []

    @staticmethod
    def find_controllable_layers(model):
        for layer in model.layers:
            if isinstance(layer, LearnedDropout):
                yield layer
            elif isinstance(layer, tf.keras.Model):
                yield from NeuralPotentialControlCallback.find_controllable_layers(layer)

    def on_train_begin(self, logs=None):
        controlled_layers = list(self.find_controllable_layers(self.model))
        if not controlled_layers:
            raise ValueError('Trying to control a network without any neural potential dropout layers')
        self.controlled_layers = controlled_layers
        self.initial_value = float(controlled_layers[0].neural_potential)
        self.integrated_error.assign(0.)
        self.log_history = []
        self.batch_count = -1

    def get_current_status(self):
        # use passthrough probability here: we want values close to 0 to be strongly represented
        probas = [tf.nn.sigmoid(layer.mask_weights) for layer in self.controlled_layers]
        mean_probas = [tf.math.reduce_mean(p) for p in probas]
        # geometric mean of passthrough probabilities
        geometric_mean = tf.pow(tf.math.reduce_prod(mean_probas), 1./len(self.controlled_layers))
        # convert to dropout probability for final use
        dropout_mean = 1 - geometric_mean

        error = self.setpoint - dropout_mean

        return CurrentStatus(dropout_mean, error)

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count%self.period:
            return

        status = self.get_current_status()
        self.integrated_error.assign_add(status.error*self.period)

        output = self.initial_value*tf.exp(self.k_p*(status.error + self.integrated_error/self.tau))

        for layer in self.controlled_layers:
            layer.neural_potential.assign(output)

        logs = {} if logs is None else logs
        logs['mean_probability'] = status.mean_probability
        logs['controller_output'] = output

        if self.log:
            self.log_history.append(LogPoint(status.mean_probability.numpy(), output.numpy()))

    def on_epoch_end(self, epoch, logs=None):
        status = self.get_current_status()

        logs = {} if logs is None else logs
        logs['mean_probability'] = status.mean_probability.numpy()
        logs['neural_potential'] = self.controlled_layers[0].neural_potential.numpy()

@flowws.add_stage_arguments
class NeuralPotentialController(flowws.Stage):
    ARGS = [
        Arg('k_p', '-k', float,
           help='Proportional constant (in log-space)'),
        Arg('tau', '-t', float,
           help='Time constant for I-control (in log-space)'),
        Arg('setpoint', '-s', float, .5,
           help='Mean passthrough probability setpoint'),
        Arg('log_history', None, bool, False,
           help='If True, retain a history in-memory'),
        Arg('period', '-p', int, 1,
           help='Period (in batches) for the controller to run'),
        Arg('disable', '-d', bool, 0,
           help='If given, disable this callback for future Train invocations'),
    ]

    def run(self, scope, storage):
        if self.arguments['disable']:
            callbacks = scope.setdefault('callbacks', [])
            new_callbacks = [c for c in callbacks if
                             not isinstance(c, NeuralPotentialControlCallback)]
            scope['callbacks'] = new_callbacks
            return

        callback = NeuralPotentialControlCallback(
            self.arguments['k_p'], self.arguments['tau'], self.arguments['setpoint'],
            self.arguments['period'], self.arguments['log_history'])
        scope.setdefault('callbacks', []).append(callback)
