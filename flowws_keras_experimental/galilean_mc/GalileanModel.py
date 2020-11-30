import functools

import flowws
from flowws import Argument as Arg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

class Model(keras.Model):
    def __init__(self, *args, galilean_steps=10, galilean_distance=1e-3,
                 galilean_batch_timescale=32, galilean_gradient_rate=.2,
                 galilean_gradient_momentum=.999, galilean_gradient_eps=1e-3,
                 gradient_descent_rate=0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.galilean_steps = galilean_steps
        self.galilean_distance = tf.Variable(galilean_distance, trainable=False, dtype=tf.float32)
        self.galilean_batch_timescale = galilean_batch_timescale
        self.galilean_gradient_rate = galilean_gradient_rate
        self.galilean_gradient_momentum = galilean_gradient_momentum
        self.galilean_gradient_eps = galilean_gradient_eps
        self.gradient_descent_rate = gradient_descent_rate

        self.rate_tracker = keras.metrics.Mean(name='gradient_step_rate')
        self.chain_delta_tracker = keras.metrics.Mean(name='chain_delta')

    def compile(self, *args, **kwargs):
        result = super().compile(*args, **kwargs)
        self._gradient_ms_accums = [
            tf.Variable(1., trainable=False) for _ in self.trainable_variables]
        return result

    def train_step(self, data):
        return tf.cond(
            tf.random.uniform([1])[0] <= self.gradient_descent_rate,
            lambda: super(Model, self).train_step(data),
            lambda: self.train_step_gmc(data))

    def train_step_gmc(self, data):
        x, y = data

        loss_fun = self.compiled_loss
        trainable_vars = self.trainable_variables

        velocities = []
        for (w, g) in zip(trainable_vars, self._gradient_ms_accums):
            v = K.random_normal(w.shape)
            scale = self.galilean_distance/K.sqrt(g + self.galilean_gradient_eps)
            velocities.append(v*scale)

        pred0 = self(x, training=True)
        loss0 = K.mean(loss_fun(y, pred0))

        def north_good(lossN, predN, velocities):
            return lossN, predN, velocities, 0

        def north_bad(loss, pred, velocities):
            # check south loss
            for (w, v) in zip(trainable_vars, velocities):
                # x0 + v (N) -> x0 - v (S)
                K.update_add(w, -2*v)

            # the base algorithm would check the gradient at x0 or
            # at x0 + v, but we are checking it at x0 - v here
            with tf.GradientTape() as tape:
                predS = self(x, training=True)
                lossS = K.mean(loss_fun(y, predS))

            south_good_f = functools.partial(south_good, loss, pred, lossS, tape, velocities)
            south_bad_f = functools.partial(south_bad, loss, pred, velocities)
            return tf.cond(
                lossS <= loss0, south_good_f, south_bad_f)

        def south_bad(loss, pred, velocities):
            # return back to initial position: x0 - v -> x0
            for (w, v) in zip(trainable_vars, velocities):
                K.update_add(w, v)

            new_v = [-v for v in velocities]

            return loss, pred, new_v, 0

        def east_good(vprimes):
            return vprimes, 1

        def west_good(vprimes):
            new_v = [-v for v in vprimes]
            return new_v, 1

        def fallback_case(velocities):
            return [-v for v in velocities], 1

        def south_good(loss, pred, lossS, tape, velocities):
            gradients = tape.gradient(lossS, trainable_vars)

            grad_mag_sq = tf.reduce_sum([tf.reduce_sum(K.square(g)) for g in gradients])
            norm = 1.0/grad_mag_sq

            vprimes = []
            for (v, g, accum) in zip(velocities, gradients, self._gradient_ms_accums):
                vflat, gflat = K.flatten(v), -K.flatten(g)
                vprime = K.reshape(2*K.sum(vflat*gflat)*norm*g, v.shape) - v
                vprimes.append(vprime)

                new_accum = (self.galilean_gradient_momentum*accum +
                    (1 - self.galilean_gradient_momentum)*tf.reduce_mean(K.square(g)))
                K.update(accum, new_accum)

            # x0 - v (S) -> x0 + vprime (E)
            for (w, v, vprime) in zip(trainable_vars, velocities, vprimes):
                K.update_add(w, v + vprime)

            predE = self(x, training=True)
            lossE = K.mean(loss_fun(y, predE))

            # x0 + vprime (E) -> x0 - vprime (W)
            for (w, v, vprime) in zip(trainable_vars, velocities, vprimes):
                K.update_add(w, -2*vprime)

            predW = self(x, training=True)
            lossW = K.mean(loss_fun(y, predW))

            go_east = tf.math.logical_and(lossE <= loss0, lossW > loss0)
            go_west = tf.math.logical_and(lossW <= loss0, lossE > loss0)

            # x0 - vprime (W) -> x0
            for (w, vprime) in zip(trainable_vars, vprimes):
                K.update_add(w, vprime)

            east_good_f = functools.partial(east_good, vprimes)
            west_good_f = functools.partial(west_good, vprimes)
            fallback_case_f = functools.partial(fallback_case, velocities)
            new_v, gradient_calcs = tf.cond(go_east, east_good_f,
                lambda: tf.cond(go_west, west_good_f, fallback_case_f))

            return loss, pred, new_v, gradient_calcs

        def loop_cond(i, loss, pred, velocities, gradient_steps):
            return i < self.galilean_steps

        def loop_body(i, loss, pred, velocities, gradient_steps):
            # set x <- x0 + v
            for (w, v) in zip(trainable_vars, velocities):
                K.update_add(w, v)

            predN = self(x, training=True)
            lossN = K.mean(loss_fun(y, predN))

            north_good_f = functools.partial(north_good, lossN, predN, velocities)
            north_bad_f = functools.partial(north_bad, loss, pred, velocities)
            (loss, pred, velocities, delta_gradients) = tf.cond(
                lossN <= loss0, north_good_f, north_bad_f)

            return (i + 1, loss, pred, velocities, gradient_steps + delta_gradients)

        (_, loss, pred, velocities, gradient_steps) = tf.while_loop(
            loop_cond, loop_body, (0, loss0, pred0, velocities, 0), swap_memory=False)

        gradient_rate = tf.cast(gradient_steps, tf.float32)/float(self.galilean_steps)
        if self.galilean_batch_timescale:
            rate_ratio = self.galilean_gradient_rate/gradient_rate
            L = float(4)
            rate_ratio = K.clip(rate_ratio, 1./L, L)
            new_distance = self.galilean_distance*tf.cast(K.pow(
                rate_ratio, 1.0/self.galilean_batch_timescale), tf.float32)
            K.update(self.galilean_distance, new_distance)

        self.rate_tracker.update_state(gradient_rate)
        chain_delta = loss - loss0
        self.chain_delta_tracker.update_state(chain_delta)

        self.compiled_metrics.update_state(y, pred)
        result = {m.name: m.result() for m in self.metrics}
        result['gradient_step_rate'] = self.rate_tracker.result()
        result['chain_delta'] = self.chain_delta_tracker.result()

        return result

    def test_step(self, data):
        x, y = data

        pred = self(x, training=False)
        self.compiled_loss(y, pred)
        self.compiled_metrics.update_state(y, pred)

        return {m.name: m.result() for m in self.metrics}

class DistanceLogger(keras.callbacks.Callback):
    def on_epoch_end(self, index, logs={}):
        logs['galilean_distance'] = K.get_value(self.model.galilean_distance)

class DummyOptimizerWrapper(object):
    def __init__(self, real_model):
        self._real_model = real_model

    def __getattr__(self, name):
        if name == 'lr':
            return self._real_model.galilean_distance
        elif name == '_real_model':
            return object.__getattribute__(self, name)
        else:
            raise NotImplementedError(name)

class DummyModelWrapper(object):
    def __init__(self, real_model):
        self._real_model = real_model

    def __getattr__(self, name):
        if name == 'optimizer':
            return DummyOptimizerWrapper(self._real_model)
        elif name == '_real_model':
            return object.__getattribute__(self, name)
        else:
            return getattr(self._real_model, name)

class ReduceStepSizeOnPlateau(keras.callbacks.ReduceLROnPlateau):
    @property
    def model(self):
        real_model = self._real_model
        wrapped_version = DummyModelWrapper(real_model)
        return wrapped_version

    @model.setter
    def model(self, value):
        self._real_model = value

@flowws.add_stage_arguments
class GalileanModel(flowws.Stage):
    ARGS = [
        Arg('steps', '-s', int, 10,
            help='Number of galilean steps to perform for each batch'),
        Arg('move_distance', '-m', float, 1e-3,
            help='Distance to move for each step'),
        Arg('log_move_distance', '-d', bool, False,
            help='If True, log the move distance'),
        Arg('tune_distance', '-t', bool, False,
            help='Auto-tune the move distance based on loss surface reflection rates'),
        Arg('gradient_descent_rate', '-g', float, 0,
            help='Fraction of steps to use normal gradient descent on'),
        Arg('reduce_distance_period', None, int, 0,
            help='Patience (in epochs) for a distance reduction method like ReduceLROnPlateau'),
        Arg('clear', '-c', bool, False,
            help='If given, remove the usage of a previous GalileanModel')
    ]

    def run(self, scope, storage):
        if self.arguments['clear']:
            scope.pop('custom_model_class')

            keep = lambda x: (
                not isinstance(x, (DistanceLogger, ReduceStepSizeOnPlateau)))
            scope['callbacks'] = list(filter(keep, scope.get('callbacks', [])))

            return

        timescale = 32 if self.arguments['tune_distance'] else 0
        ModelFun = functools.partial(
            Model, galilean_steps=self.arguments['steps'],
            galilean_distance=self.arguments['move_distance'],
            galilean_batch_timescale=timescale,
            gradient_descent_rate=self.arguments['gradient_descent_rate'],
        )
        scope['custom_model_class'] = ModelFun

        if self.arguments['log_move_distance']:
            scope.setdefault('callbacks', []).append(DistanceLogger())

        if self.arguments['reduce_distance_period']:
            callback = ReduceStepSizeOnPlateau(
                patience=self.arguments['reduce_distance_period'],
                monitor='val_loss', factor=.75, verbose=True)
            scope.setdefault('callbacks', []).append(callback)
