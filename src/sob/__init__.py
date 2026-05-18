"""sob - Self-Supervised Depth Distribution.

This package provides tools and models for self-supervised depth distribution
estimation from stereo images.
"""

__version__ = "0.1.0"

# Define public exports
__all__ = ["Trainer", "TrainingConfig"]

# Import core modules
from . import config
from . import distribution
from . import metrics
from . import projection
from . import sampling
from . import utils
from . import logger

# Import subpackages
from . import datasets
from . import networks
from . import losses

# Import commonly used classes/functions for easier access
from .trainer import Trainer
from .config import TrainingConfig
