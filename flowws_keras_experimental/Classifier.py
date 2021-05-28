import flowws
from flowws import Argument as Arg
from tensorflow import keras

from .internal import sequence

@flowws.add_stage_arguments
class Classifier(flowws.Stage):
    """Specify a simple classifier output."""

    ARGS = [
        Arg('activation', default='softmax')
    ]

    def run(self, scope, storage):
        layers = []

        layers.append(keras.layers.Dense(
            scope['num_classes'], activation=self.arguments['activation']))

        scope['output'] = sequence(scope['output'], layers)
        scope['loss'] = 'sparse_categorical_crossentropy'
        scope.setdefault('metrics', []).append('accuracy')
