import functools
import hashlib
import os
import random

import flowws
from flowws import Argument as Arg
import numpy as np
from tensorflow import keras
import PIL

AUGMENTATIONS = {}

def augmentation(f):
    AUGMENTATIONS[f.__name__] = f
    return f

@augmentation
def null(img):
    return img

AUGMENTATIONS['id'] = null

@augmentation
def scale(img, low=256, high=480):
    (w, h) = img.size
    new_size = random.randint(low, high)

    if w < h:
        (new_w, new_h) = new_size, int(h*new_size/w)
    else:
        (new_w, new_h) = int(w*new_size/h), new_size

    return img.resize((new_w, new_h))

AUGMENTATIONS['scale_256_480'] = scale

@augmentation
def crop(img, target=256):
    (w, h) = img.size
    left = random.randint(0, w - target)
    bottom = random.randint(0, h - target)
    box = (left, bottom, left + target, bottom + target)
    return img.crop(box)

AUGMENTATIONS['crop_224'] = functools.partial(crop, target=224)
AUGMENTATIONS['crop_256'] = functools.partial(crop, target=256)

@augmentation
def keras_preprocess(img):
    return keras.applications.imagenet_utils.preprocess_input(
        np.asarray(img))

@augmentation
def maybe_flip(img):
    if random.random() >= .5:
        return img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    return img

def split_filenames(directory, label_names, validation_fraction=.3, base=int(1e21)):
    train, val = [], []
    thresh = int(validation_fraction*base)
    for label_name in label_names:
        base_hash = hashlib.sha1(label_name.encode())
        dirname = os.path.join(directory, label_name)
        for fname in os.listdir(dirname):
            file_hash = base_hash.copy()
            file_hash.update(fname.encode())
            file_hash = int(file_hash.hexdigest(), base=16)%base
            if file_hash > thresh:
                train.append((label_name, fname))
            else:
                val.append((label_name, fname))
    return train, val

def batch_generator(directory_base, labeled_files, label_map, batch_size=32,
                    augmentations=[]):
    while True:
        chosen_files = random.choices(labeled_files, k=batch_size)
        xs = []
        ys = []
        for (label_name, filename) in chosen_files:
            filename = os.path.join(directory_base, label_name, filename)
            ys.append(label_map[label_name])
            x = PIL.Image.open(filename).convert('RGB')
            for aug in augmentations:
                x = aug(x)
            if np.asarray(x).ndim < 3:
                print(label_name, filename)
            xs.append(np.asarray(x))

        yield np.array(xs), np.array(ys)

@flowws.add_stage_arguments
class ImagenetDirectory(flowws.Stage):
    """Load ImageNet images from a specified directory."""

    ARGS = [
        Arg('base', '-b', str,
            help='Base directory storing images'),
        Arg('validation_fraction', '-v', float, .3,
            help='Fraction of files to be used in validation set'),
        Arg('augmentations', '-a', [str],
            help='Names of augmentations to perform on each image (use "null" for none)'),
        Arg('batch_size', None, int, 32,
            help='Batch size for training and validation'),
        Arg('train_epoch_scaling', None, float, 1.,
            help='Factor to scale the number of batches considered to be part of an epoch by (train set)'),
        Arg('val_epoch_scaling', None, float, 1.,
            help='Factor to scale the number of batches considered to be part of an epoch by (validation set)'),
        Arg('test_epoch_scaling', None, float, 1.,
            help='Factor to scale the number of batches considered to be part of an epoch by (test set)'),
    ]

    def run(self, scope, storage):
        train_dir = os.path.join(self.arguments['base'], 'train')
        test_dir = os.path.join(self.arguments['base'], 'val')

        label_names = list(sorted(os.listdir(train_dir)))
        label_map = {label: i for (i, label) in enumerate(label_names)}

        train_files, val_files = split_filenames(
            train_dir, label_names, self.arguments['validation_fraction'])

        test_files, _ = split_filenames(test_dir, label_names, -1)

        augmentation_names = (self.arguments['augmentations'] or
                              ['scale', 'crop', 'maybe_flip', 'keras_preprocess'])
        augmentations = [AUGMENTATIONS[name] for name in augmentation_names]

        train_generator = batch_generator(
            train_dir, train_files, label_map, self.arguments['batch_size'], augmentations)
        val_generator = batch_generator(
            train_dir, val_files, label_map, self.arguments['batch_size'], augmentations)
        test_generator = batch_generator(
            test_dir, test_files, label_map, self.arguments['batch_size'], augmentations)

        steps_per_epoch = int(len(train_files)//self.arguments['batch_size']*
                              self.arguments['train_epoch_scaling'])
        validation_steps = (len(val_files)//self.arguments['batch_size']*
                            self.arguments['val_epoch_scaling'])
        test_steps = (len(test_files)//self.arguments['batch_size']*
                      self.arguments['test_epoch_scaling'])

        scope['label_names'] = label_names
        scope['label_map'] = label_map
        scope['train_generator'] = train_generator
        scope['generator_train_steps'] = steps_per_epoch
        scope['validation_generator'] = val_generator
        scope['generator_val_steps'] = validation_steps
        scope['test_generator'] = test_generator
        scope['generator_test_steps'] = test_steps
        scope['loss'] = 'sparse_categorical_crossentropy'
        scope['num_classes'] = len(label_names)
