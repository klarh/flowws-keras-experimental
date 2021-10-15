"""
Implementation of gradient accumulation by github user andreped:

https://github.com/keras-team/keras/issues/3556#issuecomment-728833881
"""

import sys

import tensorflow as tf
from tensorflow.keras import backend as K

def convert(
        orig_optimizer, update_params_frequency, accumulate_sum_or_mean=True):
    if update_params_frequency < 1:
        raise ValueError('update_params_frequency must be >= 1')
    orig_get_gradients = orig_optimizer.get_gradients
    orig_get_updates = orig_optimizer.get_updates
    accumulated_iterations = K.variable(
        0, dtype='int64', name='accumulated_iterations')
    orig_optimizer.accumulated_iterations = accumulated_iterations

    def updated_get_gradients(self, loss, params):
        return self.accumulate_gradient_accumulators

    def updated_get_updates(self, loss, params):
        self.accumulate_gradient_accumulators = [
            K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        updates_accumulated_iterations = K.update_add(accumulated_iterations, 1)
        new_grads = orig_get_gradients(loss, params)
        if not accumulate_sum_or_mean:
            new_grads = [
                g/K.cast(update_params_frequency, K.dtype(g)) for g in new_grads]
        self.updated_grads = [
            K.update_add(p, g) for p, g in
            zip(self.accumulate_gradient_accumulators, new_grads)]

        def update_function():
            with tf.control_dependencies(orig_get_updates(loss, params)):
                reset_grads = [
                    K.update(p, K.zeros(K.int_shape(p), dtype=K.dtype(p)))
                    for p in self.accumulate_gradient_accumulators]
            return tf.group(*(reset_grads + [updates_accumulated_iterations]))

        def just_store_function():
            return tf.group(updates_accumulated_iterations)

        update_switch = K.equal(
            (updates_accumulated_iterations)%update_params_frequency, 0)

        with tf.control_dependencies(self.updated_grads):
            self.updates = [
                K.switch(update_switch, update_function, just_store_function)]
            return self.updates

    orig_optimizer.get_gradients = updated_get_gradients.__get__(
        orig_optimizer, type(orig_optimizer))
    orig_optimizer.get_updates = updated_get_updates.__get__(
        orig_optimizer, type(orig_optimizer))
