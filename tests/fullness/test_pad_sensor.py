"""The weighing pad's signal: settling, noise, drift, offset, arm load and quantization."""

import dataclasses
import math

import numpy as np
import pytest
from support.paths import ROOT

from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.pad import PadSensor

NOMINAL = load_fullness_config(ROOT / "configs" / "fullness.toml").pad
# An ideal pad: instant settling, no noise, offset, drift, arm load or quantization.
IDEAL = dataclasses.replace(
    NOMINAL,
    noise_sigma_g=0.0,
    offset_sigma_g=0.0,
    drift_g_per_s=0.0,
    arm_load_g=0.0,
    resolution_g=0.0,
    settle_time_constant_s=1e-6,
)


def sensor(config=IDEAL, seed=0):
    return PadSensor(config, np.random.default_rng(seed))


def test_an_ideal_pad_reads_the_true_mass():
    weighing = sensor().weigh(0.369)
    assert weighing.mass_kg == pytest.approx(0.369, abs=1e-9)
    assert weighing.stable


def test_the_signal_rises_like_a_first_order_lag():
    # After one time constant a first-order system has covered 1 - 1/e of the step, worked by hand.
    config = dataclasses.replace(IDEAL, settle_time_constant_s=0.15)
    at_one_tau = sensor(config).expected_g(0.4, 0.15)
    assert at_one_tau == pytest.approx(400.0 * (1 - math.exp(-1)), abs=1e-9)
    assert sensor(config).expected_g(0.4, 5 * 0.15) == pytest.approx(400.0 * (1 - math.exp(-5)))


def test_drift_adds_the_rate_times_the_elapsed_time():
    config = dataclasses.replace(IDEAL, drift_g_per_s=0.01)
    base = sensor(IDEAL).expected_g(0.3, 3.0)
    assert sensor(config).expected_g(0.3, 3.0) - base == pytest.approx(0.03, abs=1e-9)


def test_the_arm_load_decays_with_its_time_constant():
    config = dataclasses.replace(IDEAL, arm_load_g=3.0, arm_load_time_constant_s=0.5)
    base = sensor(IDEAL).expected_g(0.3, 0.5)
    assert sensor(config).expected_g(0.3, 0.5) - base == pytest.approx(3.0 / math.e, abs=1e-9)
    assert sensor(config).expected_g(0.3, 5.0) - sensor(IDEAL).expected_g(0.3, 5.0) < 1e-3


def test_a_weighing_is_a_zero_mean_scatter_with_the_configured_noise_after_averaging():
    config = dataclasses.replace(IDEAL, noise_sigma_g=0.5, samples_averaged=20)
    masses = sensor(config, seed=1).weigh_batch(0.3, 20000)
    # The mean of 20 samples has sigma 0.5 / sqrt(20) = 0.1118 g; a spread estimate from 20000
    # draws is good to about 0.5 %, so 5 % leaves a wide margin.
    assert masses.mean() == pytest.approx(0.3, abs=5e-6)
    assert masses.std() * 1000 == pytest.approx(0.5 / math.sqrt(20), rel=0.05)


def test_the_zero_offset_is_shared_by_one_weighings_samples_but_varies_between_weighings():
    config = dataclasses.replace(IDEAL, offset_sigma_g=0.3)
    masses = sensor(config, seed=2).weigh_batch(0.3, 20000)
    assert masses.std() * 1000 == pytest.approx(0.3, rel=0.05)  # not reduced by averaging


def test_readings_are_multiples_of_the_resolution_before_averaging():
    config = dataclasses.replace(IDEAL, resolution_g=0.1, samples_averaged=1)
    grams = sensor(config).weigh_batch(0.30047, 50) * 1000
    assert np.allclose(grams / 0.1, np.round(grams / 0.1), atol=1e-6)
    assert grams == pytest.approx(300.5, abs=1e-6)  # 300.47 rounds to the nearest 0.1 g


def test_a_pad_still_moving_is_reported_unstable():
    # Weighing too early leaves the arm's load and the settling still moving between the two halves.
    config = dataclasses.replace(
        NOMINAL, settle_wait_s=0.2, noise_sigma_g=0.0, offset_sigma_g=0.0, drift_g_per_s=0.0
    )
    assert not sensor(config).weigh(0.3).stable
    assert sensor(NOMINAL).weigh(0.3).stable


def test_the_same_seed_gives_the_same_weighing():
    first = sensor(NOMINAL, seed=7).weigh(0.3)
    second = sensor(NOMINAL, seed=7).weigh(0.3)
    assert first == second
    assert sensor(NOMINAL, seed=8).weigh(0.3) != first


def test_a_batch_of_one_matches_a_single_weighing_in_distribution_not_in_draws():
    # Same model, different draw order: only the spread has to agree.
    batch = sensor(NOMINAL, seed=3).weigh_batch(0.3, 5000)
    singles = np.array([sensor(NOMINAL, seed=s).weigh(0.3).mass_kg for s in range(400)])
    assert batch.std() == pytest.approx(singles.std(), rel=0.2)


def test_an_empty_pad_reads_about_zero_and_a_negative_mass_is_rejected():
    assert abs(sensor(NOMINAL, seed=4).weigh(0.0).mass_kg) < 0.003
    with pytest.raises(ValueError, match="mass"):
        sensor().weigh(-0.1)


def test_the_pad_file_gives_the_load_cell_a_capacity_of_one_kilogram():
    assert NOMINAL.capacity_g == 1000.0


def test_a_load_over_the_cells_capacity_reads_overloaded_and_never_as_a_light_weight():
    # A full 2 L bottle weighs about 2.1 kg: the cell saturates, it does not wrap or read low.
    weighing = sensor().weigh(2.1)
    assert weighing.overloaded and not weighing.stable
    assert weighing.mass_kg == pytest.approx(1.0)


def test_a_load_just_under_the_capacity_is_weighed_as_usual():
    weighing = sensor().weigh(0.999)
    assert not weighing.overloaded and weighing.stable
    assert weighing.mass_kg == pytest.approx(0.999, abs=1e-9)


def test_many_weighings_report_the_overload_for_each():
    masses, stable, overloaded = sensor(NOMINAL).weigh_all(2.1, 5)
    assert overloaded.all() and not stable.any() and masses == pytest.approx(1.0)
    assert not sensor(NOMINAL).weigh_all(0.4, 5)[2].any()
