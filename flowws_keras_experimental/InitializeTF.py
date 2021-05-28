import flowws
from flowws import Argument as Arg
import tensorflow as tf

@flowws.add_stage_arguments
class InitializeTF(flowws.Stage):
    """Initialize tensorflow, enabling memory growth for GPUs."""

    ARGS = [
        Arg('jit', '-j', bool, True,
            help='If True, enable JIT compilation'),
        Arg('memory_growth', '-m', bool, True,
            help='If True, enable gradual memory growth'),
    ]

    def run(self, scope, storage):
        tf.config.optimizer.set_jit(self.arguments['jit'])

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                if self.arguments['memory_growth']:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
