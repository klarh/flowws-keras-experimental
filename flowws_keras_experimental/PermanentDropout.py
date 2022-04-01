import flowws
from tensorflow import keras

class PermanentDropoutLayer(keras.layers.Dropout):
    """Dropout layer that produces noisy results even outside of training time"""
    def __init__(self, rate, **kwargs):
        super(PermanentDropoutLayer, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(x)
            x = keras.backend.dropout(x, self.rate, noise_shape)
        return x

class PermanentSpatialDropout1DLayer(keras.layers.SpatialDropout1D):
    """Dropout layer that produces noisy results even outside of training time"""
    def __init__(self, rate, **kwargs):
        super(PermanentSpatialDropout1DLayer, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(x)
            x = keras.backend.dropout(x, self.rate, noise_shape)
        return x

class PermanentDropout(flowws.Stage):
    """Make dropout layers used by later modules use test-time dropout."""

    def run(self, scope, storage):
        scope['dropout_class'] = PermanentDropoutLayer
        scope['dropout_sequence_class'] = PermanentSpatialDropout1DLayer
