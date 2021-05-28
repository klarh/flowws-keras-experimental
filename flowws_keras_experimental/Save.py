import contextlib
import hashlib
import json

import flowws
from flowws import Argument as Arg
import gtar
import keras_gtar
import numpy as np

@flowws.add_stage_arguments
class Save(flowws.Stage):
    """Save the architecture and weights of a model using keras_gtar."""

    ARGS = [
        Arg('save_model', '-s', bool, False),
        Arg('hash_size', '-c', int, 0,
            help='If given, use a hash of the workflow description for the dump filename'),
        Arg('group', '-g', str,
            help='If given, set the GTAR group ID to save with'),
        Arg('file_modifiers', '-f', [str], [],
            help='List of additional filename modifiers to use'),
    ]

    def run(self, scope, storage):

        varying = []
        if 'log_quantities' not in scope:
            history = scope['model'].history.history
            frame = scope.get('last_epoch', len(history['loss']))

            for (key, val) in history.items():
                val = np.asarray(val)
                dtype = str(val.dtype).replace('f', 'F').replace('u', 'U').replace('i', 'I')
                fmt = getattr(gtar.Format, dtype, gtar.Format.Float32)
                rec = gtar.Record('', key, str(frame), gtar.Behavior.Continuous, fmt, gtar.Resolution.Uniform)
                varying.append((rec, val))
        for (frame, quantities) in scope.get('log_quantities', []):
            for (key, val) in quantities.items():
                val = np.asarray(val)
                dtype = str(val.dtype).replace('f', 'F').replace('u', 'U').replace('i', 'I')
                fmt = getattr(gtar.Format, dtype, gtar.Format.Float32)
                rec = gtar.Record('', key, str(frame), gtar.Behavior.Continuous, fmt, gtar.Resolution.Uniform)
                varying.append((rec, val))

        with contextlib.ExitStack() as st:
            modifiers = list(self.arguments['file_modifiers'])
            if self.arguments['hash_size']:
                N = self.arguments['hash_size']
                mod = hashlib.sha1(json.dumps(
                    scope['workflow'].to_JSON()).encode()).hexdigest()[:N]
                modifiers.append(mod)

            storage_handle = st.enter_context(storage.open(
                scope.get('dump_filename', 'dump.sqlite'), 'a', modifiers, on_filesystem=True))
            traj = st.enter_context(keras_gtar.Trajectory(
                storage_handle.name, 'a', group=self.arguments.get('group', None)))

            if self.arguments['save_model']:
                traj.save(scope['model'], str(frame))

            for (rec, val) in varying:
                traj.handle.writeRecord(rec, val)

            traj.handle.writeStr('workflow.json', json.dumps(scope['workflow'].to_JSON()))
            traj.handle.writeStr('metadata.json', json.dumps(scope.get('metadata', {})))
