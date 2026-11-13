"""The tilt test: tip an unknown container with its opening high and read what the liquid does.

The torque on the grasp is the liquid's (from the statics model) plus the empty container's own,
whose weight is the pad's gross weight less the liquid. A reading names a volume; the verdict bins
only when even the largest volume the reading allows is under the threshold.
"""

import math
from dataclasses import dataclass
from functools import cache

import numpy as np

from beverage_disposal.core.coinft import CoinFT
from beverage_disposal.core.fullness import BIN, SET_ASIDE, Verdict
from beverage_disposal.core.fullness_config import RuleConfig
from beverage_disposal.core.statics import G, liquid_wrench, worst_case_torque_error
from beverage_disposal.core.tilt_config import TiltConfig

CURVE_POINTS = 121  # volumes between 0 and "all the gross weight is liquid"


@dataclass(frozen=True)
class Geometry:
    radius: float  # metres
    height: float
    grasp_height: float  # the grasp point above the container's bottom


@cache
def _liquid_torque(geometry: Geometry, volume_ml: float, tilt: float, density: float) -> float:
    """The liquid's torque alone; cached because a Monte Carlo asks for the same volumes again."""
    wrench = liquid_wrench(
        geometry.radius, geometry.height, geometry.grasp_height, volume_ml, tilt, density
    )
    return wrench.torque_nm


@cache
def _shape_error(geometry: Geometry, test) -> float:
    """The worst corner of the shape and density errors at the threshold volume, N m (cached)."""
    return worst_case_torque_error(
        geometry.radius,
        geometry.height,
        geometry.grasp_height,
        test.threshold_ml,
        math.radians(test.tilt_deg),
        test.nominal_density_g_per_ml,
        radius_error=test.radius_error_m,
        height_error=test.height_error_m,
        grasp_error=test.grasp_error_m,
        density_error=test.density_error,
        tilt_error=math.radians(test.tilt_error_deg),
    )


def total_torque(
    geometry: Geometry,
    gross_mass_kg: float,
    volume_ml: float,
    tilt: float,
    density: float,
    empty_centre_fraction: float = 0.5,
) -> float:
    """The torque on the grasp, N m, of a container holding `volume_ml` at `tilt` radians.

    `gross_mass_kg` is the whole container as weighed upright; the empty part is what is left
    after the liquid, with its centre of mass at `empty_centre_fraction` of the height.
    """
    liquid_nm = _liquid_torque(geometry, volume_ml, tilt, density)
    empty_kg = gross_mass_kg - volume_ml * density * 1e-3
    lever = -(empty_centre_fraction * geometry.height - geometry.grasp_height) * math.sin(tilt)
    return liquid_nm + empty_kg * G * lever


def torque_curve(
    geometry: Geometry,
    gross_mass_kg: float,
    tilt: float,
    density: float,
    volumes_ml: np.ndarray,
) -> np.ndarray:
    return np.array([total_torque(geometry, gross_mass_kg, v, tilt, density) for v in volumes_ml])


def uncertainty_nm(config: TiltConfig, geometry: Geometry, gross_mass_kg: float) -> float:
    """How far a reading may be from the model's torque, N m, before the verdict gives it up.

    The sensor's one-reading error to k standard deviations, plus the worst corner of the shape
    and density errors at the threshold volume, plus the empty container's unknown centre of mass.
    """
    test, sensor = config.test, config.sensor
    sigma_mnm = math.hypot(
        sensor.noise_sigma_mnm / math.sqrt(sensor.samples_averaged), sensor.offset_sigma_mnm
    )
    tilt = math.radians(test.tilt_deg)
    shape = _shape_error(geometry, test)
    empty_centre = (
        gross_mass_kg * G * test.empty_centre_error_fraction * geometry.height * math.sin(tilt)
    )
    return test.k_sigma * sigma_mnm / 1000.0 + shape + empty_centre


def _curve(config: TiltConfig, geometry: Geometry, gross_mass_kg: float):
    tilt = math.radians(config.test.tilt_deg)
    volumes = np.linspace(0.0, gross_mass_kg * 1000.0, CURVE_POINTS)
    return volumes, torque_curve(
        geometry, gross_mass_kg, tilt, config.test.nominal_density_g_per_ml, volumes
    )


def volume_upper_ml(
    config: TiltConfig, geometry: Geometry, gross_mass_kg: float, measured_torque_nm: float
) -> float | None:
    """The largest volume, mL, a reading allows; None when it lies outside the model's range."""
    volumes, curve = _curve(config, geometry, gross_mass_kg)
    slack = uncertainty_nm(config, geometry, gross_mass_kg)
    if measured_torque_nm + slack < curve[0] or measured_torque_nm - slack > curve[-1]:
        return None
    return float(np.interp(measured_torque_nm + slack, curve, volumes))


def tilt_verdict(
    config: TiltConfig, geometry: Geometry, gross_mass_kg: float, measured_torque_nm: float
) -> Verdict:
    """Bin only if the tilt test is sure the container holds under the threshold."""
    if gross_mass_kg > config.test.unknown_gross_limit_kg:
        return Verdict(SET_ASIDE, "too heavy for an unknown container", None)
    bound = volume_upper_ml(config, geometry, gross_mass_kg, measured_torque_nm)
    if bound is None:
        return Verdict(SET_ASIDE, "the reading is outside what the model allows", None)
    if bound < config.test.threshold_ml:
        return Verdict(BIN, "under the threshold even allowing for the errors", bound)
    return Verdict(SET_ASIDE, "may hold the threshold or more", bound)


def pad_only_unknown_verdict(rule: RuleConfig, gross_mass_kg: float) -> Verdict:
    """The fallback with no CoinFT: bin an unknown container only if all of it is under the limit.

    Whatever weighs under the limit cannot hold the threshold volume of the lightest drink, so this
    is never less strict than the tilt test, and it bins far fewer containers.
    """
    gross_g = gross_mass_kg * 1000.0
    if gross_g < rule.limit_g:
        return Verdict(BIN, "the whole container weighs under the limit", None)
    return Verdict(SET_ASIDE, "weighs more than the limit and was not tilt tested", None)


def simulate_measurement(
    config: TiltConfig,
    geometry: Geometry,
    volume_ml: float,
    empty_mass_kg: float,
    density: float,
    sensor: CoinFT,
    empty_centre_fraction: float = 0.5,
) -> float:
    """What the CoinFT would read tilting this container: the true torque through the sensor."""
    tilt = math.radians(config.test.tilt_deg)
    gross = empty_mass_kg + volume_ml * density * 1e-3
    true = total_torque(geometry, gross, volume_ml, tilt, density, empty_centre_fraction)
    return sensor.measure(true)
