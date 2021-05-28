import functools

import flowws
from flowws import Argument as Arg
from tensorflow import keras
from tensorflow.keras.applications import resnet

def default_clone(layer):
    return layer.__class__.from_config(layer.get_config())

def clonefun(layer, Dropout, rate):
    result = default_clone(layer)

    if (isinstance(layer, keras.layers.Activation) and
        layer.get_config()['activation'] == 'relu'):

        name = result.name + '_plus_dropout'
        return keras.Sequential([result, Dropout(rate)], name=name)
    return result

@flowws.add_stage_arguments
class ResNet(flowws.Stage):
    """Use the ResNet architecture as provided by keras."""

    ARGS = [
        Arg('size', '-s', int, 50,
            help='Size of the ResNet to build (i.e. 50, 101, etc)'),
        Arg('dropout', '-d', float, 0,
            help='Dropout probability to use (if any)'),
    ]

    def run(self, scope, storage):
        try:
            input_shape = scope['x_train'][0].shape
        except KeyError:
            input_shape = next(scope['train_generator'])[0][0].shape
        num_classes = scope['num_classes']
        Dropout = scope.get('dropout_spatial2d_class', keras.layers.Dropout)

        ResNet = getattr(resnet, 'ResNet{}'.format(self.arguments['size']))
        model = ResNet(classes=num_classes, weights=None, input_shape=input_shape)

        if self.arguments['dropout']:
            clonefun_ = functools.partial(
                clonefun, Dropout=Dropout, rate=self.arguments['dropout'])
            model = keras.models.clone_model(model, clone_function=clonefun_)

        scope['model'] = model
        scope['loss'] = 'sparse_categorical_crossentropy'
        scope.setdefault('metrics', []).append('accuracy')
