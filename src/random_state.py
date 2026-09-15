"""Shared seeding helper.

Lives in its own module so that every model can normalise an `rng` argument
without importing `model_selection`, which imports the models in turn.
"""

import numpy as np

DEFAULT_SEED = 42


def get_rng(rng=None):
    """Normalise `rng` to a Generator.

    Accepts a Generator (returned unchanged, so it keeps advancing), an integer
    seed, or None for the default seed.
    """
    if isinstance(rng, np.random.Generator):
        return rng

    return np.random.default_rng(DEFAULT_SEED if rng is None else rng)


def unseeded_rng(rng=None):
    """Like `get_rng`, but None means a fresh unseeded generator.

    Used where a helper may be called repeatedly inside a loop: seeding on None
    would hand every call an identical stream.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng()

    return np.random.default_rng(rng)
