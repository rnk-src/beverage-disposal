"""The settings of the tilt test: the CoinFT model and the test's margins."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from beverage_disposal.core.config import from_mapping


@dataclass(frozen=True)
class SensorConfig:
    """The torque sensor in the gripper pads, in millinewton metres."""

    noise_sigma_mnm: float
    offset_sigma_mnm: float
    resolution_mnm: float  # least count of one sample; 0 for none
    samples_averaged: int


@dataclass(frozen=True)
class TiltTestConfig:
    tilt_deg: float
    threshold_ml: float
    nominal_density_g_per_ml: float
    density_error: float  # g/mL either side of nominal
    radius_error_m: float
    height_error_m: float
    grasp_error_m: float
    tilt_error_deg: float
    empty_centre_error_fraction: float  # of the container's height
    k_sigma: float  # sensor errors are covered to this many standard deviations
    unknown_gross_limit_kg: float  # a heavier unknown container is set aside untested


@dataclass(frozen=True)
class TiltConfig:
    sensor: SensorConfig
    test: TiltTestConfig


def load_tilt_config(path: str | Path) -> TiltConfig:
    with open(path, "rb") as file:
        return from_mapping(TiltConfig, tomllib.load(file), "tilt")
