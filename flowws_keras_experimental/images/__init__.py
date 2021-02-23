from flowws import try_to_import

from .CIFAR10 import CIFAR10
from .CIFAR100 import CIFAR100
from .Encoder import Encoder
ImagenetDirectory = try_to_import('.ImagenetDirectory', 'ImagenetDirectory', __name__)
from .MNIST import MNIST
