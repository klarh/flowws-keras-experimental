# v0.2.0 - 2022/04/01

## Added

- Argument to force GPU usage in `InitializeTF`
- Arguments to accumulate gradients and catch keyboard interrupts in `Train`
- Argument to set monitor quantity in `Train`
- Argument to shuffle labels in `Train`
- `PermanentDropout` module

## Fixed

- Set min_delta=0 for `ReduceLROnPlateau` in `Train`

# v0.1.0 - 2021/06/01

## Added

- Basic set of modules: `InitializeTF`, `Classifier`, `MLP`, `Tensorboard`, `Train`, `Save`, `images.CIFAR100`, `images.CIFAR10`, `images.ImagenetDirectory`, `images.MNIST`, `images.Encoder`, `images.MobileNetV2`, `images.ResNet`
