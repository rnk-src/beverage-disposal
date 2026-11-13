"""Policies that must not succeed: they show the success measure is not vacuous."""

import numpy as np

from beverage_disposal.core.interfaces import Observation


class RandomController:
    """Uniform random actions in [-1, 1], repeatable by seed."""

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def reset(self, observation: Observation) -> None:
        pass

    def act(self, observation: Observation) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=4)
