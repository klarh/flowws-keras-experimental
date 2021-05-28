import flowws
from flowws import Argument as Arg
from tensorflow import keras

from .internal import sequence

@flowws.add_stage_arguments
class MLP(flowws.Stage):
    """Specify a multilayer perceptron model."""

    ARGS = [
        Arg('hidden_widths', '-w', [int], [32],
           help='Number of nodes for each hidden layer'),
        Arg('activation', '-a', str, 'relu'),
        Arg('batch_norm', '-b', bool, False,
            help='Apply batch normalization before all hidden layers'),
        Arg('output_batch_norm', None, bool, False,
            help='Apply batch normalization after each hidden layer'),
        Arg('flatten', '-f', bool, False,),
        Arg('dropout', '-d', float, 0,
            help='Apply a dropout layer with the given '
            'dropout rate after each hidden layer'),
    ]

    def run(self, scope, storage):
        input_shape = scope['x_train'][0].shape
        input_symbol = keras.layers.Input(shape=input_shape)

        Dropout = scope.get('dropout_class', keras.layers.Dropout)

        layers = []

        if self.arguments['batch_norm']:
            layers.append(keras.layers.BatchNormalization())

        if self.arguments['flatten']:
            layers.append(keras.layers.Flatten())

        for w in self.arguments['hidden_widths']:
            layers.append(keras.layers.Dense(w, activation=self.arguments['activation']))
            if self.arguments.get('output_batch_norm', False):
                layers.append(keras.layers.BatchNormalization())
            if self.arguments['dropout']:
                layers.append(Dropout(self.arguments['dropout']))

        scope['input_symbol'] = input_symbol
        scope['output'] = sequence(input_symbol, layers)
