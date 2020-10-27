import contextlib
import hashlib
import json
import random

import flowws
from flowws import Argument as Arg
import gtar
import keras_gtar
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from ..internal import sequence

NO_PARENT = '(random)'

@flowws.add_stage_arguments
class Train(flowws.Stage):
    ARGS = [
        Arg('initial_states', type=int,
           help='Number of random initializations to use, if more than 1'),
        Arg('branches', type=int, default=3,
           help='Number of branches to split for each trajectory'),
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
        Arg('seed', '-s', int),
        Arg('exponential', '-x', bool, 0,
           help='If True, branch from all possible points instead of just the initial "base" trajectories')
    ]

    def run(self, scope, storage):
        if 'seed' in self.arguments:
            s = self.arguments['seed']
            random.seed(s)
            random.seed(random.randrange(2**32))
            np.random.seed(random.randrange(2**32))
            tf.random.set_seed(random.randrange(2**32))

        model = keras.models.Model(scope['input_symbol'], scope['output'])

        scope['model'] = model

        for term in scope.get('extra_losses', []):
            model.add_loss(term)

        metrics = scope.get('metrics', [])

        model.compile(self.arguments['optimizer'], loss=scope['loss'], metrics=metrics)

        branches = self.arguments['branches']

        with contextlib.ExitStack() as context_stack:
            try:
                handle = context_stack.enter_context(storage.open(
                    scope.get('dump_filename', 'dump.sqlite'), 'r', on_filesystem=True))

                with gtar.GTAR(handle.name, 'r') as traj:
                    queue_json = traj.readStr('job_queue.json')
                    if queue_json is None:
                        raise RuntimeError()
                    job_queue = list(map(tuple, json.loads(queue_json)))

                (frames_completed, job_parent, job_grandparent, frame_index, replica) = job_queue[0]

                replica += 1
                job_queue[0] = (frames_completed, job_parent, job_grandparent, frame_index, replica)

                # when we branch from this job, start at the second
                # saved frame, rather than the first (which will be a
                # duplicate of where we are branching from)
                this_job_start = 1

                if job_parent == NO_PARENT:
                    print('Initializing a replica from scratch, but data were found')
                    frame_index = 0
                else:
                    print('Loading weights from parent ID {}, index {}'.format(job_parent, frame_index))
                    traj = context_stack.enter_context(keras_gtar.Trajectory(handle.name, 'r', job_parent))
                    model.set_weights(traj.get_weights(frame_index))
                    scope['last_epoch'] = int(traj.frames[frame_index])
                    print('restored frame index', scope['last_epoch'])

            # file doesn't exist or json file isn't found
            except (FileNotFoundError, RuntimeError):
                initial_replica = branches - self.arguments.get('initial_states', branches)
                job_queue = [(0, NO_PARENT, NO_PARENT, 0, initial_replica)]
                job_parent, frame_index = NO_PARENT, 0
                this_job_start = frames_completed = 0
                print('Initializing a replica from scratch')

        callbacks = scope.get('callbacks', [])

        callbacks.append(tfa.callbacks.TQDMProgressBar(
            show_epoch_progress=False, update_per_second=1))

        if 'early_stopping' in self.arguments:
            callbacks.append(keras.callbacks.EarlyStopping(
                patience=self.arguments['early_stopping'], monitor='val_loss'))

        if 'reduce_lr' in self.arguments:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                patience=self.arguments['reduce_lr'], monitor='val_loss', factor=.5, verbose=True))

        job_hash = hashlib.sha1(json.dumps(scope['workflow'].to_JSON()).encode()).hexdigest()[:32]

        with contextlib.ExitStack() as context_stack:
            dump_name = None
            storage_handle = context_stack.enter_context(storage.open(
                    scope.get('dump_filename', 'dump.sqlite'), 'a', on_filesystem=True))

            if self.arguments.get('dump_period', None):
                cbk = keras_gtar.GTARLogger(
                    storage_handle.name, self.arguments['dump_period'], append=True,
                    when='pre_epoch', group=job_hash)
                callbacks.append(cbk)

            model.fit(
                scope['x_train'], scope['y_train'], verbose=False, epochs=self.arguments['epochs'],
                batch_size=self.arguments['batch_size'], validation_split=self.arguments['validation_split'],
                callbacks=callbacks, initial_epoch=scope.get('last_epoch', 0))

            traj = context_stack.enter_context(gtar.GTAR(storage_handle.name, 'a'))

            metadata = dict(workflow=scope['workflow'].to_JSON())
            metadata['parent'] = job_parent
            metadata['frame_index'] = frame_index
            metadata['job_hash'] = job_hash
            traj.writeStr('{}/metadata.json'.format(job_hash), json.dumps(metadata))

            # make random initializations have themselves as parents
            if job_parent == NO_PARENT:
                job_parent = job_hash
                this_job_start = 0
            this_job_info = (frames_completed + this_job_start, job_hash, job_parent, this_job_start, 0)
            if self.arguments['exponential'] or this_job_info[0] == 0:
                job_queue.append(this_job_info)

            (frames_completed, job_parent, job_grandparent, frame_index, replica) = job_queue[0]
            if replica + 1 >= branches:
                frame_index += 1
                frames_completed += 1
                replica = 0
                job_queue.pop(0)
                if job_parent != NO_PARENT:
                    job_queue.append((frames_completed, job_parent, job_grandparent, frame_index, replica))
            job_queue.sort()

            traj.writeStr('job_queue.json', json.dumps(job_queue))
