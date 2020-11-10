import contextlib
import hashlib
import json
import random

import flowws
from flowws import Argument as Arg
import keras_gtar
import numpy as np
import tensorflow as tf
from tensorflow import keras
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
        Arg('dump_period', '-d', int),
        Arg('hash_size', '-c', int, 0,
            help='If given, use a hash of the workflow description for the dump filename'),
        Arg('seed', '-s', int),
        Arg('summarize', None, bool, False,
            help='If True, print the model summary before training'),
        Arg('verbose', None, bool, True,
            help='If True, print the training progress'),
        Arg('clean_batch_multiple', None, bool, False,
            help='If True, make the training data a clean multiple of the batch size'),
        Arg('rebuild_model', '-r', bool, False,
            help='If True, always rebuild the model when one already exists'),
    ]

    def run(self, scope, storage):
        if 'seed' in self.arguments:
            s = self.arguments['seed']
            random.seed(s)
            random.seed(random.randrange(2**32))
            np.random.seed(random.randrange(2**32))
            tf.random.set_seed(random.randrange(2**32))

        if self.arguments['clean_batch_multiple']:
            bs = self.arguments['batch_size']
            x_train = scope['x_train']
            scope['x_train'] = x_train[:len(x_train)//bs*bs]
            y_train = scope['y_train']
            scope['y_train'] = y_train[:len(y_train)//bs*bs]

        if 'model' not in scope or self.arguments['rebuild_model']:
            ModelCls = scope.get('custom_model_class', keras.models.Model)
            model = ModelCls(scope['input_symbol'], scope['output'])

            scope['model'] = model

            for term in scope.get('extra_losses', []):
                model.add_loss(term)

            metrics = scope.get('metrics', [])

            model.compile(self.arguments['optimizer'], loss=scope['loss'], metrics=metrics)

            if self.arguments['summarize']:
                model.summary()
        else:
            model = scope['model']

        callbacks = scope.get('callbacks', [])

        if 'early_stopping' in self.arguments:
            callbacks.append(keras.callbacks.EarlyStopping(
                patience=self.arguments['early_stopping'], monitor='val_loss'))

        if 'reduce_lr' in self.arguments:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                patience=self.arguments['reduce_lr'], monitor='val_loss', factor=.5, verbose=True))

        verbose = self.arguments['verbose']
        if tfa is not None and verbose:
            callbacks.append(tfa.callbacks.TQDMProgressBar(
                show_epoch_progress=False, update_per_second=1))
            verbose = False

        with contextlib.ExitStack() as context_stack:
            if self.arguments.get('dump_period', None):
                modifiers = []
                if self.arguments['hash_size']:
                    N = self.arguments['hash_size']
                    mod = hashlib.sha1(json.dumps(
                        scope['workflow'].to_JSON()).encode()).hexdigest()[:N]
                    modifiers.append(mod)

                handle = context_stack.enter_context(storage.open(
                    scope.get('dump_filename', 'dump.tar'), 'a', modifiers, on_filesystem=True))
                cbk = keras_gtar.GTARLogger(
                    handle.name, self.arguments['dump_period'], append=True, when='pre_epoch')
                callbacks.append(cbk)

            model.fit(
                scope['x_train'], scope['y_train'], verbose=verbose, epochs=self.arguments['epochs'],
                batch_size=self.arguments['batch_size'], validation_split=self.arguments['validation_split'],
                callbacks=callbacks, initial_epoch=scope.get('last_epoch', 0))

        scope['last_epoch'] = scope.get('last_epoch', 0) + len(model.history.history['loss'])
