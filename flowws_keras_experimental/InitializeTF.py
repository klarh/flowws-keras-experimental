import logging
import subprocess

import flowws
from flowws import Argument as Arg
import tensorflow as tf

logger = logging.getLogger(__name__)

@flowws.add_stage_arguments
class InitializeTF(flowws.Stage):
    """Initialize tensorflow, enabling memory growth for GPUs."""

    ARGS = [
        Arg('jit', '-j', bool, True,
            help='If True, enable JIT compilation'),
        Arg('gpu', '-g', bool, True,
            help='If False, disable GPUs'),
        Arg('memory_growth', '-m', bool, True,
            help='If True, enable gradual memory growth'),
        Arg('memory_fraction', '-f', float,
            help='If given, limit the GPU memory to the given fraction of available memory'),
    ]

    def run(self, scope, storage):
        tf.config.optimizer.set_jit(self.arguments['jit'])

        if not self.arguments['gpu']:
            tf.config.set_visible_devices([], 'GPU')

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                if self.arguments['memory_growth']:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)

                if self.arguments.get('memory_fraction', None):
                    frac = self.arguments['memory_fraction']

                    for gpu in gpus:
                        index = gpu.name.split(':')[-1]
                        command = [
                            'nvidia-smi', '--query-gpu=memory.total,memory.reserved',
                            '--format=csv,nounits,noheader', '-i', index]
                        output = subprocess.check_output(command).decode()
                        left, right = map(float, output.split(','))
                        available = left - right
                        usable = int(frac*available)

                        config = [tf.config.LogicalDeviceConfiguration(
                            memory_limit=usable)]
                        tf.config.set_logical_device_configuration(gpu, config)

                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except (RuntimeError, ValueError) as e:
                # Memory growth must be set before GPUs have been initialized
                logger.warning('Encountered an error, but continuing: {}'.format(e))
