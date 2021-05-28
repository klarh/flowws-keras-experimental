import flowws
from flowws import Argument as Arg
from tensorflow.keras import callbacks

@flowws.add_stage_arguments
class Tensorboard(flowws.Stage):
    """Specify a tensorboard dump callback."""

    ARGS = [
        Arg('histogram_period', None, int, 0,
            help='Frequency to dump histogram data'),
        Arg('write_graph', '-g', bool, True,
            help='Write the computational graph'),
        Arg('profile_batch', '-b', int, 2,
            help='Batch index to profile (0 to disable)'),
    ]

    def run(self, scope, storage):
        callback = callbacks.TensorBoard(
            histogram_freq=self.arguments['histogram_period'],
            write_graph=self.arguments['write_graph'],
            profile_batch=self.arguments['profile_batch']
        )

        scope.setdefault('callbacks', []).append(callback)
