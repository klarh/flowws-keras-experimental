import flowws
from flowws import Argument as Arg
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..internal import sequence

@flowws.add_stage_arguments
class LSTMLM(flowws.Stage):
    ARGS = [
        Arg('embedding_dimensions', '-e', int, 32,
            help='Embedding dimensionality for inputs'),
        Arg('layer_widths', '-w', [int],
            help='Number of units for each LSTM layer'),
        Arg('inner_dropout', None, float,
            help='Dropout rate to use inside LSTM module'),
        Arg('inter_dropout', None, float,
            help='Dropout rate to use after each LSTM module'),
        Arg('sequence_dropout', None, bool, False,
            help='If True, use sequence-element dropout by default'),
    ]

    def run(self, scope, storage):
        vocabulary_size = scope['vocabulary_size']
        sequence_length = scope['sequence_length']
        default_dropout = (keras.layers.SpatialDropout1D if
                           self.arguments['sequence_dropout'] else keras.layers.Dropout)
        dropout_cls = scope.get('dropout_sequence_class', default_dropout)

        inputs = last = keras.layers.Input((sequence_length,), dtype=tf.int32)
        last = keras.layers.Embedding(
            vocabulary_size, self.arguments['embedding_dimensions'])(inputs)

        for i, w in enumerate(self.arguments['layer_widths']):
            layer = keras.layers.LSTM(
                w, return_sequences=True,
                dropout=self.arguments.get('inner_dropout', 0))
            last = layer(last)

            last = keras.layers.BatchNormalization()(last)

            if 'inter_dropout' in self.arguments:
                last = dropout_cls(self.arguments['inter_dropout'])(last)

        last = keras.layers.Dense(vocabulary_size, activation='softmax')(last)

        scope['input_symbol'] = inputs
        scope['output'] = last
        scope['loss'] = 'sparse_categorical_crossentropy'
