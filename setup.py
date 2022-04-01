#!/usr/bin/env python

import os
from setuptools import setup

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

version_fname = os.path.join(THIS_DIR, 'flowws_keras_experimental', 'version.py')
with open(version_fname) as version_file:
    exec(version_file.read())

readme_fname = os.path.join(THIS_DIR, 'README.md')
with open(readme_fname) as readme_file:
    long_description = readme_file.read()

entry_points = set()
flowws_modules = []

module_names = [
    'Classifier',
    'InitializeTF',
    'MLP',
    'PermanentDropout',
    'Save',
    'Tensorboard',
    'Train',
]
for name in module_names:
    if name not in entry_points:
        flowws_modules.append('{0} = flowws_keras_experimental.{0}:{0}'.format(name))
        entry_points.add(name)
    flowws_modules.append(
        'flowws_keras_experimental.{0} = flowws_keras_experimental.{0}:{0}'.format(name))

subpkg = 'images'
module_names = [
    'CIFAR10',
    'CIFAR100',
    'Encoder',
    'ImagenetDirectory',
    'MNIST',
    'MobileNetV2',
    'ResNet',
]
for name in module_names:
    if name not in entry_points:
        flowws_modules.append('{0} = flowws_keras_experimental.{1}.{0}:{0}'.format(name, subpkg))
        entry_points.add(name)
    flowws_modules.append(
        'flowws_keras_experimental.{1}.{0} = flowws_keras_experimental.{1}.{0}:{0}'.format(name, subpkg))

subpkg = 'branch_replicas'
module_names = [
    'Train',
]
for name in module_names:
    if name not in entry_points:
        flowws_modules.append('{0} = flowws_keras_experimental.{1}.{0}:{0}'.format(name, subpkg))
        entry_points.add(name)
    flowws_modules.append(
        'flowws_keras_experimental.{1}.{0} = flowws_keras_experimental.{1}.{0}:{0}'.format(name, subpkg))

subpkg = 'galilean_mc'
module_names = [
    'GalileanModel',
]
for name in module_names:
    if name not in entry_points:
        flowws_modules.append('{0} = flowws_keras_experimental.{1}.{0}:{0}'.format(name, subpkg))
        entry_points.add(name)
    flowws_modules.append(
        'flowws_keras_experimental.{1}.{0} = flowws_keras_experimental.{1}.{0}:{0}'.format(name, subpkg))

subpkg = 'neural_potential'
module_names = [
    'NeuralPotentialController',
    'NeuralPotentialDropout',
    'PruneNeuralPotentialLayers',
]
for name in module_names:
    if name not in entry_points:
        flowws_modules.append('{0} = flowws_keras_experimental.{1}.{0}:{0}'.format(name, subpkg))
        entry_points.add(name)
    flowws_modules.append(
        'flowws_keras_experimental.{1}.{0} = flowws_keras_experimental.{1}.{0}:{0}'.format(name, subpkg))

subpkg = 'ring_replicas'
module_names = [
    'Train',
]
for name in module_names:
    if name not in entry_points:
        flowws_modules.append('{0} = flowws_keras_experimental.{1}.{0}:{0}'.format(name, subpkg))
        entry_points.add(name)
    flowws_modules.append(
        'flowws_keras_experimental.{1}.{0} = flowws_keras_experimental.{1}.{0}:{0}'.format(name, subpkg))

subpkg = 'text'
module_names = [
    'LSTMLM',
    'TransformerLM',
]
for name in module_names:
    if name not in entry_points:
        flowws_modules.append('{0} = flowws_keras_experimental.{1}.{0}:{0}'.format(name, subpkg))
        entry_points.add(name)
    flowws_modules.append(
        'flowws_keras_experimental.{1}.{0} = flowws_keras_experimental.{1}.{0}:{0}'.format(name, subpkg))

setup(name='flowws-keras-experimental',
      author='Matthew Spellings',
      author_email='mspells@vectorinstitute.ai',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      description='Stage-based scientific workflows for miscellaneous deep learning experiments',
      entry_points={
          'flowws_modules': flowws_modules,
      },
      extras_require={},
      install_requires=[
          'flowws',
      ],
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=[
          'flowws_keras_experimental',
          'flowws_keras_experimental.images',
          'flowws_keras_experimental.branch_replicas',
          'flowws_keras_experimental.galilean_mc',
          'flowws_keras_experimental.neural_potential',
          'flowws_keras_experimental.ring_replicas',
          'flowws_keras_experimental.text',
      ],
      python_requires='>=3',
      version=__version__
      )
