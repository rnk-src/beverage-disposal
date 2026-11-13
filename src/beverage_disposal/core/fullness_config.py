"""The settings of the fullness decision: the weight rule and the weighing pad model."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from beverage_disposal.core.config import from_mapping


@dataclass(frozen=True)
class RuleConfig:
    threshold_ml: float  # a container holding this much or more must be set aside
    min_density_g_per_ml: float  # the lightest drink assumed, so the threshold has a weight
    margin_g: float  # the scale's error margin, grams

    @property
    def limit_g(self) -> float:
        """Net weight (gross minus the lightest empty weight) below which a container is binned."""
        return self.threshold_ml * self.min_density_g_per_ml - self.margin_g


@dataclass(frozen=True)
class PadSensorConfig:
    """A single-point load cell through an amplifier. Grams and seconds."""

    noise_sigma_g: float  # white noise on each sample
    offset_sigma_g: float  # zero error left after taring, one draw per weighing
    drift_g_per_s: float  # slow creep of the zero
    settle_time_constant_s: float  # mechanical settling after the container is put down
    arm_load_g: float  # extra load while the arm still touches the pad
    arm_load_time_constant_s: float  # how fast that load fades as the arm lets go
    resolution_g: float  # least count of one sample; 0 for none
    sample_period_s: float
    settle_wait_s: float  # time between putting the container down and the first sample
    samples_averaged: int
    stability_limit_g: float  # the two halves of a weighing may differ by this much


@dataclass(frozen=True)
class FullnessConfig:
    rule: RuleConfig
    pad: PadSensorConfig


def load_fullness_config(path: str | Path) -> FullnessConfig:
    with open(path, "rb") as file:
        return from_mapping(FullnessConfig, tomllib.load(file), "fullness")
