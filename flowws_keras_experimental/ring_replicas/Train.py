import contextlib
import functools
import hashlib
import json
import random

import flowws
from flowws import Argument as Arg
import keras_gtar
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
try:
    import tensorflow_addons as tfa
except ImportError:
    tfa = None

@flowws.add_stage_arguments
class Train(flowws.Stage):
    ARGS = [
        Arg('optimizer', '-o', str, 'adam',
           help='optimizer to use'),
        Arg('epochs', '-e', int, 2000,
           help='Max number of epochs'),
        Arg('batch_size', '-b', int, 256,
           help='Batch size'),
        Arg('validation_split', '-v', float, .3),
        Arg('early_stopping', type=int),
        Arg('reduce_lr', type=int),
        Arg('ring_count', type=int),
        Arg('ring_k', type=float, default=1),
        Arg('ring_eps', type=float),
        Arg('dump_filename', '-f', default='dump.tar'),
        Arg('dump_period', '-d', int),
        Arg('seed', '-s', int),
        Arg('verbose', None, bool, True,
            help='If True, print the training progress'),
    ]

    @staticmethod
    def ring_name_updater(layer, i):
        cfg = layer.get_config()
        cfg['name'] = cfg['name'] + '_ring{}'.format(i)
        return layer.__class__.from_config(cfg)

    def run(self, scope, storage):
        if 'seed' in self.arguments:
            s = self.arguments['seed']
            random.seed(s)
            random.seed(random.randrange(2**32))
            np.random.seed(random.randrange(2**32))
            tf.random.set_seed(random.randrange(2**32))

        model = keras.models.Model(scope['input_symbol'], scope['output'])

        if self.arguments.get('ring_count', None):
            models = []
            for i in range(self.arguments['ring_count']):
                clone = functools.partial(self.ring_name_updater, i=i)
                models.append(keras.models.clone_model(model, scope['input_symbol'], clone))
            final_output = K.sum([m.output for m in models], axis=0)
            final_output = K.softmax(final_output)
            model = keras.models.Model(scope['input_symbol'], final_output)

            for (left, right) in zip(models, np.roll(models, -1)):
                harmonic = lambda left=left, right=right: (
                    .5*self.arguments['ring_k']*sum(
                    K.sum(K.square(l - r)) for (l, r) in zip(left.trainable_weights, right.trainable_weights)))
                model.add_loss(harmonic)

        scope['model'] = model

        for term in scope.get('extra_losses', []):
            model.add_loss(term)

        metrics = scope.get('metrics', [])

        model.compile(self.arguments['optimizer'], loss=scope['loss'], metrics=metrics)

        if self.arguments.get('ring_count', None) and self.arguments.get('ring_eps', None):
            print('randomizing ring weights')
            eps = self.arguments['ring_eps']

            names = [l.name for l in model.layers if 'ring' in l.name]
            base_values = {}
            for name in names:
                base = re.sub(r'_ring\d+$', '', name)
                layer = model.get_layer(name)
                if base in base_values:
                    for (value, tensor) in zip(base_values[base], layer.trainable_weights):
                        new_value = value*np.random.normal(loc=1, scale=eps, size=value.shape)
                        tensor.assign(value)
                else:
                    base_values[base] = [w.numpy() for w in layer.trainable_weights]
        else:
            print('not randomizing ring weights')

        callbacks = scope.get('callbacks', [])

        verbose = self.arguments['verbose']
        if tfa is not None and verbose:
            callbacks.append(tfa.callbacks.TQDMProgressBar(
                show_epoch_progress=False, update_per_second=1))
            verbose = False

        if 'early_stopping' in self.arguments:
            callbacks.append(keras.callbacks.EarlyStopping(
                patience=self.arguments['early_stopping'], monitor='val_loss'))

        if 'reduce_lr' in self.arguments:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                patience=self.arguments['reduce_lr'], monitor='val_loss', factor=.5, verbose=True))

        with contextlib.ExitStack() as context_stack:
            if self.arguments.get('dump_period', None):
                modifiers = [hashlib.sha1(json.dumps(scope['workflow'].to_JSON()).encode()).hexdigest()[:32]]
                handle = context_stack.enter_context(storage.open(
                    scope.get('dump_filename', 'dump.tar'), 'a', modifiers, on_filesystem=True))
                cbk = keras_gtar.GTARLogger(
                    handle.name, self.arguments['dump_period'], append=True, when='pre_epoch')
                callbacks.append(cbk)

            model.fit(
                scope['x_train'], scope['y_train'], verbose=verbose, epochs=self.arguments['epochs'],
                batch_size=self.arguments['batch_size'], validation_split=self.arguments['validation_split'],
                callbacks=callbacks)
