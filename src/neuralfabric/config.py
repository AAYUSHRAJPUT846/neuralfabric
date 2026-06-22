"""
Global configuration for NeuralFabric.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DTYPE = "float32"
DEFAULT_DEVICE = "cpu"
RANDOM_SEED = 42


def set_seed(seed: int) -> None:
    global RANDOM_SEED

    RANDOM_SEED = seed

    np.random.seed(seed)
