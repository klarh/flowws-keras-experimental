import flowws
from flowws import Argument as Arg
from tensorflow import keras

from .internal import sequence

@flowws.add_stage_arguments
class MLP(flowws.Stage):
    ARGS = [
        Arg('hidden_widths', '-w', [int], [32],
           help='Number of nodes for each hidden layer'),
        Arg('activation', '-a', str, 'relu'),
        Arg('batch_norm', '-b', bool, False,),
        Arg('flatten', '-f', bool, False,),
    ]

    def run(self, scope, storage):
        input_shape = scope['x_train'][0].shape
        input_symbol = keras.layers.Input(shape=input_shape)

        layers = []

        if self.arguments['batch_norm']:
            layers.append(keras.layers.BatchNormalization())

        if self.arguments['flatten']:
            layers.append(keras.layers.Flatten())

        for w in self.arguments['hidden_widths']:
            layers.append(keras.layers.Dense(w, activation=self.arguments['activation']))

        scope['input_symbol'] = input_symbol
        scope['output'] = sequence(input_symbol, layers)
