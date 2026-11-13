"""The tilt test for an unknown container: torque curve, volume bound, verdict and fallback."""

import dataclasses
import math

import numpy as np
import pytest
from support.paths import ROOT

from beverage_disposal.core.coinft import CoinFT
from beverage_disposal.core.fullness import BIN, SET_ASIDE
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.statics import G
from beverage_disposal.core.statistics import wilson_upper
from beverage_disposal.core.tilt import (
    Geometry,
    pad_only_unknown_verdict,
    simulate_measurement,
    tilt_verdict,
    torque_curve,
    total_torque,
    uncertainty_nm,
    volume_upper_ml,
)
from beverage_disposal.core.tilt_config import load_tilt_config

CONFIG = load_tilt_config(ROOT / "configs" / "tilt.toml")
FULLNESS = load_fullness_config(ROOT / "configs" / "fullness.toml")
CAN = Geometry(radius=0.033, height=0.122, grasp_height=0.10)
TILT = math.radians(CONFIG.test.tilt_deg)
EMPTY_CAN_KG = 0.014


def exact_torque(volume_ml, empty_kg=EMPTY_CAN_KG, density=1.0):
    return total_torque(CAN, empty_kg + volume_ml * density * 1e-3, volume_ml, TILT, density)


def test_the_test_file_says_sixty_degrees_and_thirty_millilitres():
    assert (CONFIG.test.tilt_deg, CONFIG.test.threshold_ml) == (60.0, 30.0)


def test_an_empty_container_loads_the_grasp_only_through_its_own_weight():
    # -m g (H/2 - grasp) sin(60 deg) with m = 14 g, worked by hand: 4.64 mN m.
    by_hand = -EMPTY_CAN_KG * G * (0.122 / 2 - 0.10) * math.sin(TILT)
    assert exact_torque(0.0) == pytest.approx(by_hand, rel=1e-9)
    assert by_hand * 1000 == pytest.approx(4.64, abs=0.01)


def test_the_torque_curve_rises_with_the_volume_so_a_reading_names_one_volume():
    volumes = np.linspace(0, 80, 41)
    curve = torque_curve(CAN, 0.014 + 0.080, TILT, 1.0, volumes)
    assert np.all(np.diff(curve) > 0)


def test_the_uncertainty_adds_the_sensor_the_shape_errors_and_the_empty_centre():
    shape = CONFIG.test
    sigma = math.hypot(0.15 / math.sqrt(20), 0.20) / 1000  # one averaged reading, N m
    gross = 0.044
    empty_centre = gross * G * shape.empty_centre_error_fraction * CAN.height * math.sin(TILT)
    total = uncertainty_nm(CONFIG, CAN, gross)
    assert total >= shape.k_sigma * sigma + empty_centre
    # Dropping the sensor term leaves the shape errors and the empty-centre term: more than 2 mN m.
    assert total * 1000 > 2.0


def test_the_volume_bound_is_the_volume_whose_torque_is_the_reading_plus_the_uncertainty():
    gross = 0.044
    reading = exact_torque(10.0)
    bound = volume_upper_ml(CONFIG, CAN, gross, reading)
    shifted = reading + uncertainty_nm(CONFIG, CAN, gross)
    curve_volumes = np.linspace(0, gross * 1000, 241)
    curve = torque_curve(CAN, gross, TILT, 1.0, curve_volumes)
    # The module's curve has 121 points and this one 241, so they agree to interpolation error.
    assert bound == pytest.approx(np.interp(shifted, curve, curve_volumes), rel=1e-4)
    assert bound > 10.0  # the bound is above the truth


def test_an_empty_can_read_without_noise_is_binned():
    verdict = tilt_verdict(CONFIG, CAN, EMPTY_CAN_KG, exact_torque(0.0))
    assert verdict.action == BIN


@pytest.mark.parametrize("volume_ml", [31.0, 35.0, 60.0])
def test_a_can_at_or_over_thirty_millilitres_read_without_noise_is_set_aside(volume_ml):
    gross = EMPTY_CAN_KG + volume_ml * 1e-3
    assert tilt_verdict(CONFIG, CAN, gross, exact_torque(volume_ml)).action == SET_ASIDE


def test_a_container_heavier_than_the_unknown_limit_is_set_aside_without_looking_at_the_torque():
    verdict = tilt_verdict(CONFIG, CAN, 0.081, exact_torque(0.0))
    assert verdict.action == SET_ASIDE and "heavy" in verdict.reason


def test_a_reading_far_outside_what_the_model_allows_is_set_aside():
    for torque in (-0.05, 0.5):
        verdict = tilt_verdict(CONFIG, CAN, EMPTY_CAN_KG, torque)
        assert verdict.action == SET_ASIDE and "model" in verdict.reason


def test_the_pad_alone_bins_an_unknown_container_only_when_its_whole_weight_is_under_the_limit():
    rule = FULLNESS.rule
    assert pad_only_unknown_verdict(rule, 0.0264).action == BIN
    assert pad_only_unknown_verdict(rule, 0.0266).action == SET_ASIDE
    assert pad_only_unknown_verdict(rule, 0.044).action == SET_ASIDE


def test_whatever_the_pad_alone_bins_holds_under_thirty_millilitres_of_the_lightest_drink():
    # Under 26.5 g gross means under 26.5 g of liquid, under 30 mL at 0.95 g/mL (27.9 mL).
    rule = FULLNESS.rule
    lightest = rule.min_density_g_per_ml
    for gross_g in np.linspace(1.0, 60.0, 119):
        if pad_only_unknown_verdict(rule, gross_g / 1000).action == BIN:
            assert gross_g / lightest < rule.threshold_ml


def run(volume_ml, density, seed, trials, empty_kg=EMPTY_CAN_KG):
    rng = np.random.default_rng(seed)
    sensor = CoinFT(CONFIG.sensor, rng)
    binned = 0
    for _ in range(trials):
        gross = empty_kg + volume_ml * density * 1e-3
        measured = simulate_measurement(CONFIG, CAN, volume_ml, empty_kg, density, sensor)
        binned += tilt_verdict(CONFIG, CAN, gross, measured).action == BIN
    return binned


@pytest.mark.medium
@pytest.mark.parametrize("density", [0.95, 1.0, 1.05])
@pytest.mark.parametrize("volume_ml", [30.0, 31.0, 35.0])
def test_nothing_of_thirty_millilitres_or_more_is_binned_over_the_drink_densities(
    volume_ml, density
):
    trials = 500
    binned = run(volume_ml, density, seed=1, trials=trials)
    assert binned == 0
    assert wilson_upper(binned, trials) < 0.01  # 0 of 500: z^2 / (500 + z^2) = 0.76 %


@pytest.mark.medium
def test_the_price_of_being_strict_is_that_only_cans_under_about_sixteen_millilitres_are_binned():
    # Measured with exact readings: 15 mL gives a bound of 29 mL (binned), 18 mL gives 32 mL (not).
    assert run(0.0, 1.0, seed=2, trials=500) / 500 > 0.95
    assert run(10.0, 1.0, seed=3, trials=500) / 500 > 0.95
    assert run(20.0, 1.0, seed=4, trials=500) / 500 < 0.05


def test_the_same_seed_gives_the_same_simulated_reading():
    def read(seed):
        sensor = CoinFT(CONFIG.sensor, np.random.default_rng(seed))
        return simulate_measurement(CONFIG, CAN, 20.0, EMPTY_CAN_KG, 1.0, sensor)

    assert read(4) == read(4)
    assert read(4) != read(5)


def test_a_looser_sensor_makes_the_bound_wider():
    loose = dataclasses.replace(
        CONFIG, sensor=dataclasses.replace(CONFIG.sensor, offset_sigma_mnm=1.0)
    )
    assert uncertainty_nm(loose, CAN, 0.044) > uncertainty_nm(CONFIG, CAN, 0.044)
