"""Monte Carlo of the weight rule: how often it bins a container of a given volume."""

import dataclasses
from dataclasses import dataclass

import numpy as np

from beverage_disposal.core.containers import ContainerRecord
from beverage_disposal.core.fullness import bin_mask
from beverage_disposal.core.fullness_config import PadSensorConfig, RuleConfig
from beverage_disposal.core.pad import PadSensor
from beverage_disposal.core.statistics import wilson_upper


@dataclass(frozen=True)
class BinResult:
    volume_ml: float
    trials: int
    binned: int
    binned_rate: float
    upper_bound: float  # Wilson 95 % upper bound on the binned rate


def run_bin(
    record: ContainerRecord,
    rule: RuleConfig,
    pad: PadSensorConfig,
    volume_ml: float,
    *,
    density: float,
    empty_mass_kg: float,
    trials: int,
    seed: int,
) -> BinResult:
    """Weigh `trials` containers of `volume_ml` of a drink of `density` g/mL and count the binned.

    The pad model weighs each one; a bottle's cap is taken as confirmed, so only the weight decides.
    """
    true_mass = empty_mass_kg + volume_ml * density / 1000.0
    masses, stable = PadSensor(pad, np.random.default_rng(seed)).weigh_many(true_mass, trials)
    net_g = (masses - record.lightest_empty_mass) * 1000.0
    binned = int((bin_mask(net_g, rule) & stable).sum())
    return BinResult(volume_ml, trials, binned, binned / trials, wilson_upper(binned, trials))


@dataclass(frozen=True)
class SweepCell:
    noise_scale: float  # multiplies the pad's noise and zero offset
    drift_scale: float  # multiplies the pad's drift
    trials: int
    binned: int
    upper_bound: float


def sweep(
    record: ContainerRecord,
    rule: RuleConfig,
    pad: PadSensorConfig,
    *,
    volume_ml: float,
    density: float,
    noise_scales: tuple[float, ...],
    drift_scales: tuple[float, ...],
    trials: int,
    seed: int,
    empty_mass_kg: float | None = None,
) -> list[SweepCell]:
    """Binned counts for every noise and drift scale; noise is the outer loop."""
    tare = record.lightest_empty_mass if empty_mass_kg is None else empty_mass_kg
    cells = []
    for noise in noise_scales:
        for drift in drift_scales:
            scaled = dataclasses.replace(
                pad,
                noise_sigma_g=pad.noise_sigma_g * noise,
                offset_sigma_g=pad.offset_sigma_g * noise,
                drift_g_per_s=pad.drift_g_per_s * drift,
            )
            result = run_bin(
                record,
                rule,
                scaled,
                volume_ml,
                density=density,
                empty_mass_kg=tare,
                trials=trials,
                seed=seed,
            )
            cells.append(SweepCell(noise, drift, trials, result.binned, result.upper_bound))
    return cells


def first_breakdown(cells: list[SweepCell], limit: float) -> SweepCell | None:
    """The first cell, in sweep order, whose upper bound is above `limit`, or None."""
    return next((cell for cell in cells if cell.upper_bound > limit), None)
