from __future__ import annotations

import numpy as np

from neuralfabric import config


def test_default_configuration() -> None:
    assert config.DEFAULT_DTYPE == "float32"
    assert config.DEFAULT_DEVICE == "cpu"
    assert config.RANDOM_SEED == 42


def test_set_seed_updates_global_seed() -> None:
    config.set_seed(123)

    assert config.RANDOM_SEED == 123


def test_set_seed_makes_numpy_deterministic() -> None:
    config.set_seed(42)

    first = np.random.rand()

    config.set_seed(42)

    second = np.random.rand()

    assert first == second
