"""The CoinFT torque sensor model: noise, zero offset, least count and averaging."""

import dataclasses
import math

import numpy as np
import pytest
from support.paths import ROOT

from beverage_disposal.core.coinft import CoinFT
from beverage_disposal.core.tilt_config import load_tilt_config

SENSOR = load_tilt_config(ROOT / "configs" / "tilt.toml").sensor
IDEAL = dataclasses.replace(SENSOR, noise_sigma_mnm=0.0, offset_sigma_mnm=0.0, resolution_mnm=0.0)


def sensor(config, seed=0):
    return CoinFT(config, np.random.default_rng(seed))


def test_an_ideal_sensor_reads_the_true_torque():
    assert sensor(IDEAL).measure(0.0123) == pytest.approx(0.0123, abs=1e-12)


def test_the_averaged_noise_has_the_configured_spread():
    config = dataclasses.replace(IDEAL, noise_sigma_mnm=0.15, samples_averaged=20)
    readings = sensor(config, 1).measure_batch(0.01, 20000)
    # 0.15 mN m / sqrt(20) = 0.0335 mN m; a spread from 20000 draws is good to 0.5 %.
    assert readings.std() * 1000 == pytest.approx(0.15 / math.sqrt(20), rel=0.05)
    # The mean of 20000 readings is good to about 1.4e-6 N m (the zero offset), so 1e-5 is 7 sigma.
    assert readings.mean() == pytest.approx(0.01, abs=1e-5)


def test_the_zero_offset_is_not_averaged_away():
    config = dataclasses.replace(IDEAL, offset_sigma_mnm=0.2, samples_averaged=20)
    readings = sensor(config, 2).measure_batch(0.0, 20000)
    assert readings.std() * 1000 == pytest.approx(0.2, rel=0.05)


def test_one_sample_is_rounded_to_the_least_count():
    config = dataclasses.replace(IDEAL, resolution_mnm=0.05, samples_averaged=1)
    reading_mnm = sensor(config).measure(0.01234) * 1000
    assert reading_mnm == pytest.approx(12.35, abs=1e-9)  # 12.34 rounds to the nearest 0.05


def test_the_same_seed_gives_the_same_reading():
    assert sensor(SENSOR, 5).measure(0.01) == sensor(SENSOR, 5).measure(0.01)
    assert sensor(SENSOR, 5).measure(0.01) != sensor(SENSOR, 6).measure(0.01)
