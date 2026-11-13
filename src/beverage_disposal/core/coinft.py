"""A model of the CoinFT torque sensor in the gripper pads."""

import numpy as np

from beverage_disposal.core.tilt_config import SensorConfig


class CoinFT:
    """Torque readings: the true torque plus a per-reading zero offset and white noise on each
    sample, rounded to the least count and averaged. Torques in newton metres."""

    def __init__(self, config: SensorConfig, rng: np.random.Generator):
        self._config = config
        self._rng = rng

    def measure(self, true_torque_nm: float) -> float:
        return float(self.measure_batch(true_torque_nm, 1)[0])

    def measure_batch(self, true_torque_nm: float, n: int) -> np.ndarray:
        """`n` independent readings of the same true torque, in newton metres."""
        c = self._config
        samples = (
            true_torque_nm * 1000.0
            + self._rng.normal(0.0, c.offset_sigma_mnm, size=(n, 1))
            + self._rng.normal(0.0, c.noise_sigma_mnm, size=(n, c.samples_averaged))
        )
        if c.resolution_mnm > 0:
            samples = np.round(samples / c.resolution_mnm) * c.resolution_mnm
        return samples.mean(axis=1) / 1000.0
