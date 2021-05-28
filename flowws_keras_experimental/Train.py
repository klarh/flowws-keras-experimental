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

OPTIMIZER_MAP = dict(
    adadelta='Adadelta',
    adam='Adam',
    rmsprop='RMSprop',
    sgd='SGD',
)

@flowws.add_stage_arguments
class Train(flowws.Stage):
    """Build a model and perform some number of training steps.

    Training will proceed for the model and dataset that have been
    specified in previous stages.

    """
    ARGS = [
        Arg('optimizer', '-o', str, 'adam',
           help='optimizer to use'),
        Arg('optimizer_kwargs', None, [(str, eval)], [],
            help='Keyword arguments to pass to optimizer'),
        Arg('epochs', '-e', int, 2000,
           help='Max number of epochs'),
        Arg('batch_size', '-b', int, 256,
           help='Batch size'),
        Arg('validation_split', '-v', float, .3),
        Arg('early_stopping', type=int),
        Arg('early_stopping_best', None, type=bool,
            help='If True, restore the best weights at the end of early stopping'),
        Arg('reduce_lr', type=int),
        Arg('reduce_lr_factor', None, float, .5,
            help='Factor to scale learning rate with reduce_lr enabled'),
        Arg('dump_period', '-d', int),
        Arg('hash_size', '-c', int, 0,
            help='If given, use a hash of the workflow description for the dump filename'),
        Arg('seed', '-s', int),
        Arg('summarize', None, bool, False,
            help='If True, print the model summary before training'),
        Arg('recompile', None, bool, False,
            help='If True, always compile the model in this stage'),
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
        Arg('disable_tqdm', None, bool, False,
            help='If True, don\'t use tqdm to display a progress bar'),
        Arg('use_multiprocessing', None, bool, True,
            help='If True, use multiprocessing with generators'),
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

        metrics = scope.get('metrics', [])

        if self.arguments['optimizer_kwargs']:
            optimizer_cls = getattr(
                keras.optimizers, OPTIMIZER_MAP[self.arguments['optimizer']])
            optimizer = optimizer_cls(**dict(self.arguments['optimizer_kwargs']))
        else:
            optimizer = self.arguments['optimizer']

        should_compile = self.arguments['recompile']

        if 'model' not in scope or self.arguments['rebuild_model']:
            ModelCls = scope.get('custom_model_class', keras.models.Model)
            model = ModelCls(scope['input_symbol'], scope['output'])

            scope['model'] = model

            for term in scope.get('extra_losses', []):
                model.add_loss(term)

            should_compile = True
        else:
            model = scope['model']

        if self.arguments['summarize']:
            model.summary()

        if should_compile:
            model.compile(optimizer, loss=scope['loss'], metrics=metrics)

        callbacks = list(scope.get('callbacks', []))

        if 'early_stopping' in self.arguments:
            callbacks.append(keras.callbacks.EarlyStopping(
                patience=self.arguments['early_stopping'], monitor='val_loss',
                restore_best_weights=self.arguments.get('early_stopping_best', False)))

        if 'reduce_lr' in self.arguments:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                patience=self.arguments['reduce_lr'], monitor='val_loss',
                factor=self.arguments['reduce_lr_factor'], verbose=True))

        verbose = self.arguments['verbose']
        if tfa is not None and verbose and not self.arguments['disable_tqdm']:
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
                kwargs['use_multiprocessing'] = self.arguments['use_multiprocessing']

                if 'validation_generator' in scope:
                    kwargs['validation_data'] = scope['validation_generator']
                    kwargs['validation_steps'] = (self.arguments.get('generator_val_steps', None) or
                                                  scope.get('generator_val_steps', None))
            else:
                args.extend([scope['x_train'], scope['y_train']])
                kwargs['batch_size'] = self.arguments['batch_size']
                kwargs['validation_split'] = self.arguments['validation_split']

                if 'validation_data' in scope:
                    kwargs['validation_data'] = scope['validation_data']

            model.fit(*args, **kwargs)

        if self.arguments['epochs']:
            current_epoch = scope['last_epoch'] = scope['last_epoch'] + len(model.history.history['loss'])
            log_quantities = scope.setdefault('log_quantities', [])
            log_quantities.append((current_epoch, model.history.history))
