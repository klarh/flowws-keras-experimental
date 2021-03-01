import flowws
from flowws import Argument as Arg
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..internal import sequence

class PositionalEncoding(keras.layers.Layer):
    def build(self, input_shape):
        sequence_length = input_shape[-2]
        d_model = input_shape[-1]
        self.encoding = self.positional_encoding(sequence_length, d_model)

    def call(self, input_):
        return input_ + self.encoding

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    @classmethod
    def positional_encoding(cls, position, d_model):
        angle_rads = cls.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

keras.utils.get_custom_objects()['PositionalEncoding'] = PositionalEncoding

class MultiHeadSelfAttention(keras.layers.MultiHeadAttention):
    def build(self, input_shape):
        self.sequence_length = input_shape[1]
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        attention_mask = (np.arange(self.sequence_length).reshape((-1, 1)) >=
                          np.arange(self.sequence_length).reshape((1, -1)))
        kwargs['attention_mask'] = attention_mask
        return super().call(inputs, inputs, **kwargs)

keras.utils.get_custom_objects()['MultiHeadSelfAttention'] = MultiHeadSelfAttention

@flowws.add_stage_arguments
class TransformerLM(flowws.Stage):
    ARGS = [
        Arg('num_blocks', '-b', int, 6,
           help='Number of transformer blocks to apply'),
        Arg('num_heads', None, int, 8,
           help='Number of heads to use for multi-head self-attention'),
        Arg('dimensions', '-d', int, 128,
           help='Working dimension size for the model'),
        Arg('feedforward_dimensions', '-f', int,
           help='Dimension of the hidden layer of the feedforward neural network'),
        Arg('position_encoding', '-p', bool, True,
           help='Enable positional encoding'),
        Arg('inter_block_separable_conv', None, bool, False,
           help='If True, use separable convolutions before attention inside transformer blocks'),
        Arg('convolution_width', None, int, 3,
           help='Kernel size for convolutions (if used)'),
        Arg('convolution_activation', None, str,
           help='Activation function for convolution (if used)'),
        Arg('attention_dropout', None, bool, False,
           help='Enable dropout inside attention layers'),
        Arg('feedforward_dropout', None, bool, False,
           help='Enable dropout inside feedforward projection layers'),
        Arg('inter_block_dropout', None, bool, False,
           help='Enable dropout between blocks'),
        Arg('linear_dropout', None, bool, 0,
           help='Enable dropout at the end of the network'),
        Arg('reuse_dropout_object', None, bool, False,
           help='Use the same Dropout object between blocks'),
        Arg('dropout', None, float, .5,
            help='Dropout rate to use when enabled'),
        Arg('sequence_dropout', None, bool, False,
            help='If True, use sequence-element dropout by default'),
    ]

    def run(self, scope, storage):
        vocabulary_size = scope['vocabulary_size']
        sequence_length = scope['sequence_length']
        dimensions = self.arguments['dimensions']
        ff_dim = self.arguments.get('feedforward_dimensions', dimensions)
        head_dimensions = dimensions//self.arguments['num_heads']
        dropout_rate = self.arguments['dropout']
        default_dropout = (keras.layers.SpatialDropout1D if
                           self.arguments['sequence_dropout'] else keras.layers.Dropout)
        dropout_cls = scope.get('dropout_sequence_class', default_dropout)

        if self.arguments['reuse_dropout_object']:
            dropout_ob = dropout_cls(dropout_rate)
            dropout_cls = lambda x: dropout_ob

        inputs = last = keras.layers.Input((sequence_length,), dtype=tf.int32)
        last = keras.layers.Embedding(vocabulary_size, dimensions)(inputs)
        if self.arguments['position_encoding']:
            last = PositionalEncoding()(last)

        attention_dropout = dropout_rate if self.arguments['attention_dropout'] else 0

        for i in range(self.arguments['num_blocks']):
            pre_block = last

            att = MultiHeadSelfAttention(
                self.arguments['num_heads'], head_dimensions, head_dimensions,
                attention_axes=1, dropout=attention_dropout)
            last = att(last)
            last = keras.layers.Add()([pre_block, last])
            last = keras.layers.LayerNormalization()(last)

            mid_block = last
            last = keras.layers.Dense(ff_dim, activation='relu')(last)
            if self.arguments['feedforward_dropout']:
                last = dropout_cls(dropout_rate)(last)
            last = keras.layers.Dense(dimensions)(last)
            last = keras.layers.Add()([mid_block, last])

            if self.arguments['inter_block_separable_conv']:
                conv = keras.layers.SeparableConv1D(
                    dimensions, self.arguments['convolution_width'], padding='causal',
                    activation=self.arguments.get('convolution_activation', None),
                    )
                last = conv(last)

            if self.arguments['inter_block_dropout']:
                last = dropout_cls(dropout_rate)(last)

        if self.arguments['linear_dropout']:
            last = dropout_cls(dropout_rate)(last)
        last = keras.layers.Dense(vocabulary_size, activation='softmax')(last)

        scope['input_symbol'] = inputs
        scope['output'] = last
        scope['loss'] = 'sparse_categorical_crossentropy'
