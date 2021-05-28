import flowws
from flowws import Argument as Arg
import numpy as np
from tensorflow import keras

from ..internal import sequence

@flowws.add_stage_arguments
class Encoder(flowws.Stage):
    """Specify simple CNN-based encoder layers for image-type data."""

    ARGS = [
        Arg('convolution_widths', '-c', [int],
           help='Number of channels to build for each convolution layer'),
        Arg('dropout', '-d', float, .125,
           help='Dropout frequency for encoder'),
        Arg('final_pool', '-p', str,
            help='Type of final pooling to apply for encoding, if any: one of "max", "avg"'),
        Arg('separable_convolutions', '-s', bool, False,
            help='If True, use separable convolutions'),
        Arg('normalize_before_activation', '-n', bool, False,
            help='If True, place batch normalization before the post-conv activation'),
        Arg('encoding_dropout', None, float, 0,
            help='Dropout rate for final encoding output, if any'),
    ]

    def run(self, scope, storage):
        try:
            input_shape = scope['x_train'][0].shape
        except KeyError:
            input_shape = next(scope['train_generator'])[0][0].shape
        conv_dropout = self.arguments['dropout']
        conv_widths = self.arguments['convolution_widths']
        ConvLayer = (keras.layers.SeparableConv2D if self.arguments['separable_convolutions']
                     else keras.layers.Conv2D)

        Dropout = scope.get('dropout_class', keras.layers.Dropout)
        SpatialDropout2D = scope.get('dropout_spatial2d_class', keras.layers.SpatialDropout2D)

        input_symbol = keras.layers.Input(shape=input_shape)
        current_size = input_shape[-2]

        padded_size = 2**(int(np.ceil(np.log2(current_size))))
        pad_radius = (padded_size - current_size)//2
        current_size = padded_size

        layers = []
        layers.append(keras.layers.BatchNormalization(input_shape=input_shape))
        layers.append(keras.layers.ZeroPadding2D((pad_radius, pad_radius)))
        for w in conv_widths:
            activation = None if self.arguments['normalize_before_activation'] else 'relu'
            layers.append(ConvLayer(
                w, kernel_size=3, activation=activation, padding='same'))
            layers.append(keras.layers.BatchNormalization())
            if self.arguments['normalize_before_activation']:
                layers.append(keras.layers.Activation('relu'))
            layers.append(keras.layers.AveragePooling2D(pool_size=(2, 2)))
            current_size //= 2
            if conv_dropout:
                layers.append(SpatialDropout2D(conv_dropout))

        last_size = current_size

        final_pool = self.arguments.get('final_pool', None)
        if final_pool == 'avg':
            layers.append(keras.layers.GlobalAveragePooling2D())
            current_size = 1
        elif final_pool == 'max':
            layers.append(keras.layers.GlobalMaxPooling2D())
            current_size = 1
        elif final_pool:
            raise NotImplementedError(final_pool)

        layers.append(keras.layers.Flatten())
        current_size = current_size**2*conv_widths[-1]

        if self.arguments['encoding_dropout']:
            layers.append(Dropout(self.arguments['encoding_dropout']))

        encoded = sequence(input_symbol, layers)

        scope['pad_radius'] = pad_radius
        scope['last_size'] = last_size
        scope['convolution_widths'] = conv_widths
        scope['input_symbol'] = input_symbol
        scope['output'] = encoded
        scope['encoding_dim'] = current_size
