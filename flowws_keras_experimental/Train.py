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
        Arg('generator_train_steps', None, int, None,
            help='Number of steps to use as an epoch for training from a generator'),
        Arg('generator_val_steps', None, int, None,
            help='Number of steps to use as an epoch for evaluation from a generator'),
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

        callbacks = list(scope.get('callbacks', []))

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

            initial_epoch = scope.setdefault('last_epoch', 0)
            total_epochs = initial_epoch + self.arguments['epochs']

            args = []
            kwargs = dict(
                verbose=verbose,
                epochs=total_epochs,
                callbacks=callbacks,
                initial_epoch=initial_epoch
            )

            if 'train_generator' in scope:
                args.append(scope['train_generator'])
                kwargs['steps_per_epoch'] = (self.arguments.get('generator_train_steps', None) or
                                             scope.get('generator_train_steps', None))

                if 'validation_generator' in scope:
                    kwargs['validation_data'] = scope['validation_generator']
                    kwargs['validation_steps'] = (self.arguments.get('generator_val_steps', None) or
                                                  scope.get('generator_val_steps', None))
            else:
                args.extend([scope['x_train'], scope['y_train']])
                kwargs['batch_size'] = self.arguments['batch_size']
                kwargs['validation_split'] = self.arguments['validation_split']

            model.fit(*args, **kwargs)

        current_epoch = scope['last_epoch'] = scope['last_epoch'] + len(model.history.history['loss'])
        log_quantities = scope.setdefault('log_quantities', [])
        log_quantities.append((current_epoch, model.history.history))
