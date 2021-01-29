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

class Pruner:
    INPUT_CLASS_FUNCTIONS = {}
    OUTPUT_CLASS_FUNCTIONS = {}
    PASSTHROUGH_CLASS_FUNCTIONS = {}

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

    @classmethod
    def register_passthrough(cls, registered_cls):
        def result(fn):
            cls.PASSTHROUGH_CLASS_FUNCTIONS[registered_cls] = fn
            return fn
        return result

    @classmethod
    def prune_input(cls, layer, mask):
        for (target_class, target_f) in cls.INPUT_CLASS_FUNCTIONS.items():
            if isinstance(layer.source, target_class):
                layer = target_f(layer, mask)
                return layer

    @classmethod
    def prune_output(cls, layer, mask):
        for (target_class, target_f) in cls.OUTPUT_CLASS_FUNCTIONS.items():
            if isinstance(layer.source, target_class):
                layer = target_f(layer, mask)
                return layer

    @classmethod
    def prune_passthrough(cls, layer, mask):
        for (target_class, target_f) in cls.PASSTHROUGH_CLASS_FUNCTIONS.items():
            if isinstance(layer.source, target_class):
                layer = target_f(layer, mask)
                return layer

@Pruner.register_passthrough(keras.layers.BatchNormalization)
def prune_batchnorm(desc, mask):
    new_weights = []
    for w in desc.weights:
        w = w[mask]
        new_weights.append(w)
    return desc._replace(weights=new_weights)

@Pruner.register_input(keras.layers.Dense)
def prune_dense(desc, mask):
    new_weights = []
    for w in desc.weights:
        if w.ndim == 2:
            w = w[mask]
        new_weights.append(w)
    return desc._replace(weights=new_weights)

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
    new_json['config']['units'] = len(mask)
    return desc._replace(json=new_json, weights=new_weights)

@Pruner.register_input(keras.layers.Conv2D)
def prune_conv2d(desc, mask):
    new_weights = []
    for w in desc.weights:
        if w.ndim == 4:
            w = w[..., mask, :]
        new_weights.append(w)
    return desc._replace(weights=new_weights)

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
    new_json['config']['filters'] = len(mask)
    return desc._replace(json=new_json, weights=new_weights)

@flowws.add_stage_arguments
class PruneNeuralPotentialLayers(flowws.Stage):
    ARGS = [
    ]

    def run(self, scope, storage):
        model = scope['model']
        model_json = json.loads(model.to_json())

        layer_descriptions = [
            LayerDescription(layer, layer_json, layer.get_weights())
            for (layer, layer_json) in zip(model.layers, model_json['config']['layers'])]
        last_mask_indices = last_projection_layer = last_output_shape = None
        mask_name_remaps = {}
        for i in range(len(layer_descriptions)):
            if isinstance(layer_descriptions[i].source, LearnedDropout):
                json_desc = layer_descriptions[i].json
                weights = layer_descriptions[i].weights[0]
                probas = tf.math.sigmoid(weights).numpy()
                sampled_mask = np.random.uniform(0, 1, size=probas.shape) < probas
                last_mask_indices = np.where(sampled_mask)[0]

                mask_name_remaps[json_desc['name']] = json_desc['inbound_nodes'][0][0][0]

                if last_projection_layer is not None:
                    desc = Pruner.prune_output(layer_descriptions[last_projection_layer], last_mask_indices)
                    assert desc is not None
                    layer_descriptions[last_projection_layer] = desc

                    for j in range(last_projection_layer + 1, i):
                        layer_descriptions[j] = (
                            Pruner.prune_passthrough(layer_descriptions[j], last_mask_indices) or
                            layer_descriptions[j])
            elif last_mask_indices is not None:
                if isinstance(layer_descriptions[i].source, keras.layers.Flatten):
                    batch_shape = last_output_shape[1:]
                    indices = np.arange(np.product(batch_shape)).reshape(batch_shape)
                    indices = indices[..., last_mask_indices]
                    last_mask_indices = indices.reshape(-1)

                desc = Pruner.prune_input(layer_descriptions[i], last_mask_indices)
                if desc is not None:
                    layer_descriptions[i] = desc
                    last_mask_indices = None

            new_json = copy.deepcopy(layer_descriptions[i].json)
            for layer_instance in new_json['inbound_nodes']:
                for input_description in layer_instance:
                    input_description[0] = mask_name_remaps.get(input_description[0], input_description[0])
            layer_descriptions[i] = layer_descriptions[i]._replace(json=new_json)

            if Pruner.prune_output(layer_descriptions[i], [0]) is not None:
                last_projection_layer = i

            last_output_shape = layer_descriptions[i].source.output_shape

        final_descriptions = [desc for desc in layer_descriptions if not isinstance(desc.source, LearnedDropout)]

        model_json['config']['layers'] = [desc.json for desc in final_descriptions]
        weights = sum([desc.weights for desc in final_descriptions], [])

        new_model = keras.models.model_from_json(json.dumps(model_json))

        new_model.set_weights(weights)
        new_model.compile(optimizer=model.optimizer, loss=scope['loss'], metrics=scope.get('metrics', []))
        scope['model'] = new_model
