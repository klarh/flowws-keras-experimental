from flowws import try_to_import

from . import images

from .Classifier import Classifier
from .InitializeTF import InitializeTF
from .MLP import MLP
Save = try_to_import('.Save', 'Save', __name__)
from .Tensorboard import Tensorboard
Train = try_to_import('.Train', 'Train', __name__)
