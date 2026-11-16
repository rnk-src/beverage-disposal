"""A model of the weighing pad: what a load cell would read, with its faults."""

import math
from dataclasses import dataclass

import numpy as np

from beverage_disposal.core.fullness_config import PadSensorConfig


@dataclass(frozen=True)
class Weighing:
    mass_kg: float  # the settled reading
    stable: bool  # whether the reading had stopped moving
    overloaded: bool = False  # the load was over the cell's capacity: the reading means nothing


class PadSensor:
    """Samples a container's weight after it is put down and returns the averaged reading.

    Each sample is the true mass seen through a first-order lag, plus the arm's fading load, the
    zero drift, a per-weighing zero offset and white noise, rounded to the least count.
    """

    def __init__(self, config: PadSensorConfig, rng: np.random.Generator):
        self._config = config
        self._rng = rng

    def expected_g(self, true_mass_kg: float, t: float) -> float:
        """The noise-free reading in grams `t` seconds after the container is put down."""
        c = self._config
        settled = true_mass_kg * 1000.0 * (1.0 - math.exp(-t / c.settle_time_constant_s))
        arm = c.arm_load_g * math.exp(-t / c.arm_load_time_constant_s)
        return settled + arm + c.drift_g_per_s * t

    def weigh(self, true_mass_kg: float) -> Weighing:
        masses, stable, overloaded = self.weigh_all(true_mass_kg, 1)
        return Weighing(float(masses[0]), bool(stable[0]), bool(overloaded[0]))

    def weigh_batch(self, true_mass_kg: float, n: int) -> np.ndarray:
        """`n` independent weighings of the same container: the settled masses in kilograms."""
        return self.weigh_many(true_mass_kg, n)[0]

    def weigh_many(self, true_mass_kg: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        """`n` independent weighings: settled masses (kg) and whether each had stopped moving."""
        return self.weigh_all(true_mass_kg, n)[:2]

    def weigh_all(self, true_mass_kg: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`n` independent weighings: masses (kg), whether each settled, whether each overloaded.

        A load over the cell's capacity reads as the capacity, unsettled and overloaded: a
        saturated cell must never pass for a light container.
        """
        if true_mass_kg < 0:
            raise ValueError(f"mass must not be negative, got {true_mass_kg} kg")
        c = self._config
        if true_mass_kg * 1000.0 > c.capacity_g:
            return np.full(n, c.capacity_g / 1000.0), np.zeros(n, bool), np.ones(n, bool)
        times = c.settle_wait_s + c.sample_period_s * np.arange(c.samples_averaged)
        ideal = np.array([self.expected_g(true_mass_kg, t) for t in times])
        readings = (
            ideal
            + self._rng.normal(0.0, c.offset_sigma_g, size=(n, 1))
            + self._rng.normal(0.0, c.noise_sigma_g, size=(n, len(times)))
        )
        if c.resolution_g > 0:
            readings = np.round(readings / c.resolution_g) * c.resolution_g
        half = max(1, len(times) // 2)
        moving = np.abs(readings[:, :half].mean(axis=1) - readings[:, -half:].mean(axis=1))
        return readings.mean(axis=1) / 1000.0, moving <= c.stability_limit_g, np.zeros(n, bool)
