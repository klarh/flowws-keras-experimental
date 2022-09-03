import collections
import contextlib
import hashlib
import json
import random
import re
import time

import flowws
from flowws import Argument as Arg
import gtar
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

def generator_label_shuffler(seed, gen):
    rng = np.random.default_rng(seed)
    for batch in gen:
        rng.shuffle(batch[-1])
        yield batch

class SigtermException(Exception):
    @classmethod
    def handle(cls, signum, frame):
        raise cls()

    @classmethod
    def register(cls):
        import signal
        signal.signal(signal.SIGTERM, cls.handle)

class SuppressExceptionScope(contextlib.AbstractContextManager):
    def __init__(self, scope, exctype, label):
        self._scope = scope
        self._exctype = exctype
        self._label = label

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        result = exctype is not None and issubclass(exctype, self._exctype)
        if result:
            print('Caught {}, exiting for {}'.format(str(exctype), self._label))
            self._scope.setdefault('exit_reason', []).append(self._label)
        return result

class GTARLog(keras.callbacks.Callback):
    def __init__(self, filename, buffer_size=8, group='gtar_log'):
        self.buffers = collections.defaultdict(list)
        self.handle = gtar.GTAR(filename, 'a')
        self.buffer_size = buffer_size
        self.group = group
        self._last_frame = '0'

    def flush(self, key, frame):
        val = self.buffers.pop(key)
        if not len(val):
            return
        val = np.asarray(val)
        dtype = str(val.dtype).replace('f', 'F').replace('u', 'U').replace('i', 'I')
        fmt = getattr(gtar.Format, dtype, gtar.Format.Float32)
        rec = gtar.Record(self.group, key, frame, gtar.Behavior.Continuous, fmt,
                          gtar.Resolution.Uniform)
        self.handle.writeRecord(rec, val)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for (k, v) in logs.items():
            self.buffers[k].append(v)

        frame = self._last_frame = str(epoch)
        if any(len(v) >= self.buffer_size for v in self.buffers.values()):
            for k in list(self.buffers):
                self.flush(k, frame)

    def on_train_end(self, logs=None):
        for k in list(self.buffers):
            self.flush(k, self._last_frame)
        self.handle.close()

class TimedBackupAndRestore(keras.callbacks.BackupAndRestore):
    def __init__(self, time_limit, *args,
                 train_generator=None, train_generator_steps=None,
                 validation_generator=None, validation_generator_steps=None,
                 **kwargs):
        self.time_limit = time_limit
        self._last_runtime = 0
        self._duration = self.parse_time(self.time_limit)
        self._train_generator = train_generator
        self._train_generator_steps = train_generator_steps
        assert (train_generator is None) or (train_generator_steps is not None)
        self._validation_generator = validation_generator
        self._validation_generator_steps = validation_generator_steps
        assert (validation_generator is None) or (validation_generator_steps is not None)
        self._loaded_epoch = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def parse_time(t):
        """Parse a duration string (i.e. 18h3m) into a number of seconds"""
        seconds = dict(h=60*60, m=60, s=1)
        time_regex = re.compile(r'(?P<count>\d+)(?P<unit>[hms])')
        components = re.findall(time_regex, t)

        matched_pieces = ''.join([''.join(pair) for pair in components])
        assert matched_pieces == t, 'Failed to fully parse "{}"'.format(t)

        result = 0
        for (count, unit) in components:
            result += int(count)*seconds[unit]
        return result

    def on_epoch_begin(self, epoch, logs=None):
        if self._loaded_epoch is None:
            self._loaded_epoch = epoch
        if self._train_generator is not None:
            for _ in range(epoch*self._train_generator_steps):
                next(self._train_generator)
            self._train_generator = None
        if self._validation_generator is not None:
            for _ in range(epoch*self._validation_generator_steps):
                next(self._validation_generator)
            self._validation_generator = None

    def on_epoch_end(self, *args, **kwargs):
        current_time = time.time()
        if current_time > self._last_runtime + self._duration:
            super().on_epoch_end(*args, **kwargs)
            self._last_runtime = current_time

    def get_config(self):
        result = super().get_config()
        result['time_limit'] = self.time_limit
        return result

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
        Arg('accumulate_gradients', None, int,
            help='Number of batches over which to accumulate gradients before applying'),
        Arg('catch_keyboard_interrupt', None, bool, False,
            help='If True, catch keyboard interrupts and continue to the next stage'),
        Arg('monitor_quantity', None, str, 'val_loss',
            help='Quantity to monitor for reduce_lr and early_stopping'),
        Arg('shuffle_labels', None, bool, False,
            help='If True, shuffle labels for training'),
        Arg('checkpoint_dir', None, str,
            help='If given, save and restore model checkpoints at the given location'),
        Arg('checkpoint_duration', None, str, '10m',
            help='Time duration for model checkpointing, if enabled'),
        Arg('catch_sigterm', None, bool, False,
            help='If True, catch sigterm events and continue to the next stage'),
        Arg('terminate_on_nan', None, bool, False,
            help='If True, terminate training on nan loss'),
        Arg('gtar_log_period', None, int,
            help='Number of epochs to buffer for logging quantities via GTAR'),
        Arg('gtar_log_modifiers', None, [str],
            help='Filename modifiers for live logging of quantities via GTAR'),
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
        should_compile |= 'accumulate_gradients' in self.arguments

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
            if isinstance(optimizer, str):
                optimizer = keras.optimizers.get(optimizer)

            if 'accumulate_gradients' in self.arguments:
                from .accumulate_gradients import convert
                convert(optimizer, self.arguments['accumulate_gradients'])

            compile_kwargs = scope.get('compile_kwargs', {})

            model.compile(optimizer, loss=scope['loss'], metrics=metrics, **compile_kwargs)

        callbacks = list(scope.get('callbacks', []))

        if 'early_stopping' in self.arguments:
            callbacks.append(keras.callbacks.EarlyStopping(
                patience=self.arguments['early_stopping'],
                monitor=self.arguments['monitor_quantity'],
                restore_best_weights=self.arguments.get('early_stopping_best', False)))

        if 'reduce_lr' in self.arguments:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                patience=self.arguments['reduce_lr'],
                monitor=self.arguments['monitor_quantity'],
                factor=self.arguments['reduce_lr_factor'],
                verbose=True, min_delta=0))

        restore_callback = None
        if 'checkpoint_dir' in self.arguments:
            kwargs = {}
            if 'train_generator' in scope:
                kwargs['train_generator'] = scope['train_generator']
                kwargs['train_generator_steps'] = (
                    self.arguments.get('generator_train_steps', None) or
                    scope.get('generator_train_steps', None))
                if 'validation_generator' in scope:
                    kwargs['validation_generator'] = scope['validation_generator']
                    kwargs['validation_generator_steps'] = (
                        self.arguments.get('generator_val_steps', None) or
                        scope.get('generator_val_steps', None))
            restore_callback = TimedBackupAndRestore(
                self.arguments['checkpoint_duration'],
                self.arguments['checkpoint_dir'], **kwargs)
            callbacks.append(restore_callback)

        if self.arguments['terminate_on_nan']:
            callbacks.append(keras.callbacks.TerminateOnNaN())

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
                train_gen = scope['train_generator']
                if self.arguments['shuffle_labels']:
                    train_gen = generator_label_shuffler(
                        self.arguments.get('seed', 13), train_gen)
                args.append(train_gen)
                kwargs['steps_per_epoch'] = (self.arguments.get('generator_train_steps', None) or
                                             scope.get('generator_train_steps', None))
                kwargs['use_multiprocessing'] = self.arguments['use_multiprocessing']

                if 'validation_generator' in scope:
                    val_gen = scope['validation_generator']
                    if self.arguments['shuffle_labels']:
                        val_gen = generator_label_shuffler(
                            self.arguments.get('seed', 13), val_gen)
                    kwargs['validation_data'] = val_gen
                    kwargs['validation_steps'] = (self.arguments.get('generator_val_steps', None) or
                                                  scope.get('generator_val_steps', None))
            else:
                labels = scope['y_train']
                if self.arguments['shuffle_labels']:
                    labels = labels.copy()
                    np.random.shuffle(labels)
                args.extend([scope['x_train'], labels])
                kwargs['batch_size'] = self.arguments['batch_size']
                kwargs['validation_split'] = self.arguments['validation_split']

                if 'validation_data' in scope:
                    kwargs['validation_data'] = scope['validation_data']

            with contextlib.ExitStack() as st:
                if self.arguments.get('gtar_log_period', None):
                    mods = self.arguments['gtar_log_modifiers']
                    storage_handle = st.enter_context(storage.open(
                        scope.get('dump_filename', 'dump.sqlite'), 'a',
                        mods, on_filesystem=True))
                    filename = storage_handle.name
                    callback = GTARLog(filename, self.arguments['gtar_log_period'])
                    callbacks.append(callback)

                if self.arguments['catch_keyboard_interrupt']:
                    st.enter_context(SuppressExceptionScope(
                        scope, KeyboardInterrupt, 'catch_keyboard_interrupt'))

                if self.arguments['catch_sigterm']:
                    SigtermException.register()
                    st.enter_context(SuppressExceptionScope(
                        scope, SigtermException, 'catch_sigterm'))

                model.fit(*args, **kwargs)

        if self.arguments['epochs']:
            last_epoch = scope['last_epoch']
            if restore_callback is not None:
                last_epoch = restore_callback._loaded_epoch
            current_epoch = last_epoch + len(model.history.history['loss'])
            scope['last_epoch'] = current_epoch
            log_quantities = scope.setdefault('log_quantities', [])
            log_quantities.append((current_epoch, model.history.history))
