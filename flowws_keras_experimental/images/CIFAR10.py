import flowws
import numpy as np
from tensorflow import keras

class CIFAR10(flowws.Stage):
    """Use the CIFAR10 dataset from keras."""

    def run(self, scope, storage):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

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
