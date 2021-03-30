import copy
from collections import namedtuple
import json

import flowws
from flowws import Argument as Arg
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .NeuralPotentialDropout import LearnedDropout

LayerDescription = namedtuple('LayerDescription', ['source', 'json', 'weights'])

def identity(x):
    return x

def find_learnable_dropout_layers(model):
    for layer in model.layers:
        if isinstance(layer, keras.models.Model):
            yield from find_learnable_dropout_layers(layer)
        elif isinstance(layer, LearnedDropout):
            yield layer

def get_input_parent_map(model):
    result = {}
    result[model.layers[0]] = model

    for layer in model.layers:
        if isinstance(layer, keras.Model):
            result.update(get_input_parent_map(layer))
    return result

def get_desc(layer):
    layer_json = layer.get_config()
    return LayerDescription(layer, layer_json, layer.get_weights())

class Pruner:
    INPUT_CLASS_FUNCTIONS = {}
    PASS_INPUT_CLASS_FUNCTIONS = {}
    OUTPUT_CLASS_FUNCTIONS = {}
    PASS_OUTPUT_CLASS_FUNCTIONS = {}

    @classmethod
    def register_pass_input(cls, registered_cls):
        def result(fn):
            cls.PASS_INPUT_CLASS_FUNCTIONS[registered_cls] = fn
            # make sure that superclass handlers don't get invoked
            # if they have an associated input function registered
            cls.INPUT_CLASS_FUNCTIONS[registered_cls] = lambda *args, **kwargs: None
            return fn
        return result

    @classmethod
    def register_pass_output(cls, registered_cls):
        def result(fn):
            cls.PASS_OUTPUT_CLASS_FUNCTIONS[registered_cls] = fn
            # make sure that superclass handlers don't get invoked
            # if they have an associated output function registered
            cls.OUTPUT_CLASS_FUNCTIONS[registered_cls] = lambda *args, **kwargs: None
            return fn
        return result

    @classmethod
    def register_input(cls, registered_cls):
        def result(fn):
            cls.INPUT_CLASS_FUNCTIONS[registered_cls] = fn
            return fn
        return result

    @classmethod
    def register_output(cls, registered_cls):
        def result(fn):
            cls.OUTPUT_CLASS_FUNCTIONS[registered_cls] = fn
            return fn
        return result

    @staticmethod
    def _prune(functions, layer, mask):
        try:
            target_f = functions[layer.source.__class__]
            return target_f(layer, mask)
        except KeyError:
            pass

        for (target_class, target_f) in functions.items():
            if isinstance(layer.source, target_class):
                layer = target_f(layer, mask)
                return layer

    @classmethod
    def prune_pass_input(cls, layer, mask):
        return Pruner._prune(cls.PASS_INPUT_CLASS_FUNCTIONS, layer, mask)

    @classmethod
    def prune_pass_output(cls, layer, mask):
        return Pruner._prune(cls.PASS_OUTPUT_CLASS_FUNCTIONS, layer, mask)

    @classmethod
    def prune_input(cls, layer, mask):
        return Pruner._prune(cls.INPUT_CLASS_FUNCTIONS, layer, mask)

    @classmethod
    def prune_output(cls, layer, mask):
        return Pruner._prune(cls.OUTPUT_CLASS_FUNCTIONS, layer, mask)

    @classmethod
    def mask_children_downward(cls, layer, mask, new_descriptions):
        children = [new_descriptions.get(n.layer, get_desc(n.layer))
                   for n in layer.outbound_nodes]

        for child in children:
            new_desc = cls.prune_input(child, mask)
            if new_desc is not None:
                (new_desc, _) = new_desc
                new_descriptions[new_desc.source] = new_desc
                continue

            new_desc = cls.prune_pass_input(child, mask)
            if new_desc is not None:
                (new_desc, new_mask) = new_desc
                new_descriptions[new_desc.source] = new_desc
                cls.mask_children_downward(new_desc.source, new_mask, new_descriptions)
            else:
                cls.mask_children_downward(child.source, mask, new_descriptions)

    @classmethod
    def mask_parents_upward(cls, layer, mask, new_descriptions, parent_map):
        all_layers = [layer]
        if layer in parent_map:
            all_layers.append(parent_map[layer])

        parents = []
        for layer in all_layers:
            for n in layer.inbound_nodes:
                if isinstance(n.inbound_layers, list):
                    parents.extend([new_descriptions.get(layer, get_desc(layer))
                                    for layer in n.inbound_layers])
                else:
                    parents.append(new_descriptions.get(
                        n.inbound_layers, get_desc(n.inbound_layers)))

        for parent in parents:
            new_desc = cls.prune_output(parent, mask)
            if new_desc is not None:
                (new_desc, _) = new_desc
                new_descriptions[new_desc.source] = new_desc
                continue

            new_desc = cls.prune_pass_output(parent, mask)
            if new_desc is not None:
                (new_desc, new_mask) = new_desc
                new_descriptions[new_desc.source] = new_desc
                cls.mask_parents_upward(new_desc.source, new_mask, new_descriptions, parent_map)
            else:
                cls.mask_parents_upward(parent.source, mask, new_descriptions, parent_map)

@Pruner.register_pass_input(keras.layers.BatchNormalization)
@Pruner.register_pass_output(keras.layers.BatchNormalization)
def prune_batchnorm(desc, mask):
    new_weights = []
    for w in desc.weights:
        w = w[mask]
        new_weights.append(w)
    return desc._replace(weights=new_weights), mask

@Pruner.register_pass_input(keras.layers.DepthwiseConv2D)
def prune_depthconv2d(desc, mask):
    new_weights = []
    kernel_shape = None
    for w in desc.weights:
        if w.ndim == 4:
            kernel_shape = w.shape
            w = w[..., mask, :]
        elif w.ndim == 1:
            w = w.reshape((desc.source.depth_multiplier, -1))
            w = w[:, mask]
            w = w.reshape((-1,))
        new_weights.append(w)
    new_mask = np.arange(kernel_shape[-1]*kernel_shape[-2]).reshape(kernel_shape[-2:])
    new_mask = new_mask[mask, :]
    new_mask = new_mask.reshape((-1,))
    return desc._replace(weights=new_weights), new_mask

@Pruner.register_pass_input(keras.layers.Flatten)
def prune_flatten(desc, mask):
    batch_shape = desc.source.input_shape[1:]
    indices = np.arange(np.product(batch_shape)).reshape(batch_shape)
    indices = indices[..., mask]
    new_mask = indices.reshape((-1,))
    return desc, new_mask

@Pruner.register_input(keras.layers.Dense)
def prune_dense(desc, mask):
    new_weights = []
    for w in desc.weights:
        if w.ndim == 2:
            w = w[mask]
        new_weights.append(w)
    return desc._replace(weights=new_weights), mask

@Pruner.register_output(keras.layers.InputLayer)
def prune_input(desc, mask):
    new_json = copy.deepcopy(desc.json)
    shape = list(new_json['batch_input_shape'])
    shape[-1] = len(mask)
    new_json['batch_input_shape'] = shape
    return desc._replace(json=new_json), mask

@Pruner.register_output(keras.layers.Dense)
def prune_dense(desc, mask):
    new_json = copy.deepcopy(desc.json)
    new_weights = []
    for w in desc.weights:
        if w.ndim == 2:
            w = w[:, mask]
        else:
            w = w[mask]
        new_weights.append(w)
    new_json['units'] = len(mask)
    return desc._replace(json=new_json, weights=new_weights), mask

@Pruner.register_input(keras.layers.Conv2D)
def prune_conv2d(desc, mask):
    new_weights = []
    for w in desc.weights:
        if w.ndim == 4:
            w = w[..., mask, :]
        new_weights.append(w)
    return desc._replace(weights=new_weights), mask

@Pruner.register_output(keras.layers.Conv2D)
def prune_conv2d(desc, mask):
    new_json = copy.deepcopy(desc.json)
    new_weights = []
    for w in desc.weights:
        if w.ndim == 4:
            w = w[..., mask]
        else:
            w = w[mask]
        new_weights.append(w)
    new_json['filters'] = len(mask)
    return desc._replace(json=new_json, weights=new_weights), mask

@flowws.add_stage_arguments
class PruneNeuralPotentialLayers(flowws.Stage):
    ARGS = [
        Arg('summarize', None, bool, False,
            help='If True, print the model summary after pruning'),
    ]

    def run(self, scope, storage):
        model = scope['model']
        parent_map = get_input_parent_map(model)

        to_skip, new_descriptions = [], {}
        for layer in find_learnable_dropout_layers(model):
            weights = layer.get_weights()[0]
            probas = tf.math.sigmoid(weights).numpy()
            sampled_mask = np.random.uniform(0, 1, size=probas.shape) < probas
            mask = np.where(sampled_mask)[0]

            to_skip.append(layer)
            Pruner.mask_children_downward(layer, mask, new_descriptions)
            Pruner.mask_parents_upward(layer, mask, new_descriptions, parent_map)

        new_layer_weights = {}
        nullified_indices = {layer: i for (i, layer) in enumerate(to_skip)}

        def clonefun(layer):
            result = None

            if layer in to_skip:
                name = 'nullified_learnable_dropout_{}'.format(nullified_indices[layer])
                result = keras.layers.Lambda(identity, name=name)
            if layer in new_descriptions:
                desc = new_descriptions[layer]
                result = layer.__class__.from_config(desc.json)
                new_layer_weights[result] = desc.weights
            elif isinstance(layer, keras.models.Model):
                old_input_layer = keras.models.Model.layers.fget(layer)[0]
                new_input_tensor = None
                if old_input_layer in new_descriptions:
                    new_input_desc = new_descriptions[old_input_layer]
                    new_input_layer = old_input_layer.__class__.from_config(new_input_desc.json)
                    new_input_tensor = new_input_layer.output
                result = keras.models.clone_model(layer, new_input_tensor, clone_function=clonefun)

            if result is not None:
                return result

            return layer.__class__.from_config(layer.get_config())

        new_model = keras.models.clone_model(model, clone_function=clonefun)

        for (layer, weights) in new_layer_weights.items():
            layer.set_weights(weights)

        if self.arguments['summarize']:
            new_model.summary()

        scope['model'] = new_model
