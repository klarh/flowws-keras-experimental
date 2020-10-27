import functools

import flowws
from flowws import Argument as Arg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

loss_tracker = keras.metrics.Mean(name='loss')
rate_tracker = keras.metrics.Mean(name='gradient_step_rate')
chain_delta_tracker = keras.metrics.Mean(name='chain_delta')

class Model(keras.Model):
    def __init__(self, *args, galilean_steps=10, galilean_distance=1e-3,
                 galilean_batch_timescale=32, galilean_gradient_rate=.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.galilean_steps = galilean_steps
        self.galilean_distance = tf.Variable(galilean_distance, trainable=False, dtype=tf.float32)
        self.galilean_batch_timescale = galilean_batch_timescale
        self.galilean_gradient_rate = galilean_gradient_rate

    def train_step(self, data):
        x, y = data

        loss_fun = self.compiled_loss
        trainable_vars = self.trainable_variables

        velocities = [K.random_normal(w.shape)*self.galilean_distance
                      for w in trainable_vars]

        pred0 = self(x, training=True)
        loss0 = K.mean(loss_fun(y, pred0))

        def north_good(lossN, velocities):
            return lossN, velocities, 0

        def north_bad(loss, velocities):
            # check south loss
            for (w, v) in zip(trainable_vars, velocities):
                # x0 + v (N) -> x0 - v (S)
                K.update_add(w, -2*v)

            # the base algorithm would check the gradient at x0 or
            # at x0 + v, but we are checking it at x0 - v here
            with tf.GradientTape() as tape:
                predS = self(x, training=True)
                lossS = K.mean(loss_fun(y, predS))

            south_good_f = functools.partial(south_good, loss, lossS, tape, velocities)
            south_bad_f = functools.partial(south_bad, loss, velocities)
            return tf.cond(
                lossS <= loss0, south_good_f, south_bad_f)

        def south_bad(loss, velocities):
            # return back to initial position: x0 - v -> x0
            for (w, v) in zip(trainable_vars, velocities):
                K.update_add(w, v)

            new_v = [-v for v in velocities]

            return loss, new_v, 0

        def east_good(vprimes):
            return vprimes, 1

        def west_good(vprimes):
            new_v = [-v for v in vprimes]
            return new_v, 1

        def fallback_case(velocities):
            return [-v for v in velocities], 1

        def south_good(loss, lossS, tape, velocities):
            gradients = tape.gradient(lossS, trainable_vars)

            vprimes = []
            for (v, g) in zip(velocities, gradients):
                vflat, gflat = K.flatten(v), K.flatten(g)
                n = gflat/K.sum(K.square(gflat))
                vprime = v - K.reshape(2*K.sum(vflat*n)*n, v.shape)
                vprimes.append(vprime)

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

            return loss, new_v, gradient_calcs

        def loop_cond(i, loss, velocities, gradient_steps):
            return i < self.galilean_steps

        def loop_body(i, loss, velocities, gradient_steps):
            # set x <- x0 + v
            for (w, v) in zip(trainable_vars, velocities):
                K.update_add(w, v)

            predN = self(x, training=True)
            lossN = K.mean(loss_fun(y, predN))

            north_good_f = functools.partial(north_good, lossN, velocities)
            north_bad_f = functools.partial(north_bad, loss, velocities)
            (loss, velocities, delta_gradients) = tf.cond(
                lossN <= loss0, north_good_f, north_bad_f)

            return (i + 1, loss, velocities, gradient_steps + delta_gradients)

        (_, loss, velocities, gradient_steps) = tf.while_loop(
            loop_cond, loop_body, (0, loss0, velocities, 0), swap_memory=False)

        gradient_rate = tf.cast(gradient_steps, tf.float32)/float(self.galilean_steps)
        if self.galilean_batch_timescale:
            rate_ratio = self.galilean_gradient_rate/gradient_rate
            L = float(4)
            rate_ratio = K.clip(rate_ratio, 1./L, L)
            new_distance = self.galilean_distance*tf.cast(K.pow(
                rate_ratio, 1.0/self.galilean_batch_timescale), tf.float32)
            K.update(self.galilean_distance, new_distance)

        loss_tracker.update_state(loss)
        rate_tracker.update_state(gradient_rate)

        chain_delta = loss - loss0
        chain_delta_tracker.update_state(chain_delta)

        return dict(loss=loss_tracker.result(),
                    gradient_step_rate=rate_tracker.result(),
                    chain_delta=chain_delta_tracker.result(),
                    )

class DistanceLogger(keras.callbacks.Callback):
    def on_epoch_end(self, index, logs={}):
        logs['galilean_distance'] = K.get_value(self.model.galilean_distance)

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
            help='Auto-tune the move distance'),
    ]

    def run(self, scope, storage):
        timescale = 32 if self.arguments['tune_distance'] else 0
        ModelFun = functools.partial(
            Model, galilean_steps=self.arguments['steps'],
            galilean_distance=self.arguments['move_distance'],
            galilean_batch_timescale=timescale)
        scope['custom_model_class'] = ModelFun

        if self.arguments['log_move_distance']:
            scope.setdefault('callbacks', []).append(DistanceLogger())
