import flowws
from flowws import Argument as Arg
import numpy as np
from tensorflow import keras

@flowws.add_stage_arguments
class CIFAR100(flowws.Stage):
    """Use the CIFAR100 dataset from keras."""

    ARGS = [
        Arg('coarse_labels', '-c', bool, False,
            help='If True, use 20 coarse-grained labels instead of 100 fine-grained labels'),
    ]

    def run(self, scope, storage):
        label_mode = 'coarse' if self.arguments['coarse_labels'] else 'fine'
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode=label_mode)

        num_classes = len(np.unique(y_train))

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        input_shape = x_train[0].shape

        scope['x_train'] = x_train
        scope['x_test'] = x_test
        scope['y_train'] = y_train
        scope['y_test'] = y_test
        scope['num_classes'] = num_classes
