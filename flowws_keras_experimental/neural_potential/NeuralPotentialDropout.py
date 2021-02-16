import collections
import functools
import math

import flowws
from flowws import Argument as Arg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

def dropout_mask_op_builder(batch_dims=1, reduction='sum'):
    g1_mean_axes = tuple(range(batch_dims))

    if reduction == 'sum':
        gradient_reduce_fun = K.sum
    elif reduction == 'mean':
        gradient_reduce_fun = K.mean
    else:
        raise NotImplementedError(reduction)

    @tf.custom_gradient
    def simple_dropout_mask(weights, x):
        probas = tf.math.sigmoid(weights)

        samp = tf.random.uniform(tf.shape(x), maxval=1.)
        mask = tf.cast(samp <= probas, tf.float32)
        result = mask*x
        dmask_dweights = probas*tf.math.sigmoid(-weights)

        def grad(dy):
            g1 = dy*x*dmask_dweights
            g1 = gradient_reduce_fun(g1, axis=g1_mean_axes)
            g2 = dy*mask
            return g1, g2

        return result, grad
    return simple_dropout_mask

class LearnedDropout(keras.layers.Layer):
    def __init__(self, neural_potential, initial_dropout=.5, batch_dims=1, mask_dims=1, **kwargs):
        super().__init__(**kwargs)
        self.neural_potential = tf.Variable(neural_potential, trainable=False, dtype=tf.float32)
        self.initial_dropout = initial_dropout
        self.batch_dims = batch_dims
        self.mask_dims = mask_dims

        self.maskfun = dropout_mask_op_builder(self.batch_dims)

    def build(self, input_shape):
        mask_shape = input_shape[-self.mask_dims:]
        assert not any(v is None for v in mask_shape)

        initial_weight = -math.log(1./(1 - self.initial_dropout) - 1)

        self.mask_weights = self.add_weight(
            'mask_weights', mask_shape,
            initializer=keras.initializers.RandomNormal(initial_weight),
            trainable=True)

        return super().build(input_shape)

    def call(self, input_):
        mask_weights = self.mask_weights
        for i in range(self.mask_dims, K.ndim(input_)):
            mask_weights = K.expand_dims(mask_weights, 0)

        self.add_loss(self.neural_potential*tf.reduce_sum(tf.math.sigmoid(self.mask_weights)))

        return self.maskfun(mask_weights, input_)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['neural_potential'] = float(self.neural_potential)
        config['initial_dropout'] = self.initial_dropout
        config['mask_dims'] = self.mask_dims
        config['batch_dims'] = self.batch_dims
        return config

keras.utils.get_custom_objects()['LearnedDropout'] = LearnedDropout

class LearnedSequenceDropout(LearnedDropout):
    def __init__(self, *args, spatial_mask=False, **kwargs):
        self.spatial_mask = spatial_mask
        kwargs.setdefault('batch_dims', 1 if spatial_mask else 2)
        kwargs.setdefault('mask_dims', 2 if spatial_mask else 1)
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config['spatial_mask'] = self.spatial_mask
        return config

keras.utils.get_custom_objects()['LearnedSequenceDropout'] = LearnedSequenceDropout

class LearnedSpatialDropout2D(LearnedDropout):
    def __init__(self, *args, spatial_mask=False, **kwargs):
        self.spatial_mask = spatial_mask
        kwargs.setdefault('batch_dims', 1 if spatial_mask else 3)
        kwargs.setdefault('mask_dims', 3 if spatial_mask else 1)
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config['spatial_mask'] = self.spatial_mask
        return config

keras.utils.get_custom_objects()['LearnedSpatialDropout2D'] = LearnedSpatialDropout2D

@flowws.add_stage_arguments
class NeuralPotentialDropout(flowws.Stage):
    ARGS = [
        Arg('reset', '-r', bool, False,
            help='If given, first reset (clear) dropout configuration'),
        Arg('mu', '-m', float,
            help='Neural potential to use for layers'),
        Arg('spatial_mask', '-s', bool, False,
            help='If True, learn a spatial mask for spatial dropout'),
    ]

    def run(self, scope, storage):
        mu = self.arguments['mu']

        if self.arguments['reset']:
            scope.pop('dropout_class', None)
            scope.pop('dropout_spatial2d_class', None)

        layer_dropout = functools.partial(LearnedDropout, mu)
        sequence_dropout = functools.partial(
            LearnedSequenceDropout, mu, spatial_mask=self.arguments['spatial_mask'])
        spatial_dropout = functools.partial(
            LearnedSpatialDropout2D, mu, spatial_mask=self.arguments['spatial_mask'])

        scope['dropout_class'] = layer_dropout
        scope['dropout_sequence_class'] = sequence_dropout
        scope['dropout_spatial2d_class'] = spatial_dropout
