"""The simulated camera: unbiased on average, honest intervals, a hidden level, cap and base."""

import dataclasses
import math

import numpy as np
import pytest
from scipy.stats import chi2
from support.paths import ROOT
from test_perception import BOTTLE, SHAPE

from beverage_disposal.core.bottles import level_for_volume_m, total_height_m
from beverage_disposal.core.perception import load_perception_noise, perceive
from beverage_disposal.core.policy import load_policy_config
from beverage_disposal.core.statistics import wilson_interval

PATH = ROOT / "configs" / "perception.toml"
NOISE = load_perception_noise(PATH)
N = 20_000
TRUE = {
    "body_diameter": SHAPE.body_diameter,
    "total_height": total_height_m(SHAPE),
    "shoulder_height": SHAPE.shoulder_height,
    "neck_diameter": SHAPE.neck_diameter,
    "neck_height": SHAPE.neck_height,
}
LABELLED = dataclasses.replace(BOTTLE, shape=dataclasses.replace(SHAPE, label_mass=0.001))


def looks(n=N, bottle=BOTTLE, volume_ml=100.0, noise=NOISE, seed=0, **options):
    rng = np.random.default_rng(seed)
    return [perceive(bottle, volume_ml, noise, rng, **options) for _ in range(n)]


def test_the_noise_file_holds_the_assumed_figures():
    assert (NOISE.dim_sigma_m, NOISE.dim_bias_max_m, NOISE.interval_k) == (0.002, 0.001, 4.0)
    assert NOISE.label_band == (0.30, 0.70) and NOISE.level_dropout == 0.10
    with pytest.raises(ValueError):
        load_perception_noise(ROOT / "configs" / "policy.toml")


@pytest.mark.parametrize("name", list(TRUE))
def test_a_measured_length_is_right_on_average_and_spread_as_configured(name):
    middles = np.array([sum(getattr(seen.dims, name)) / 2 for seen in looks()])
    errors = middles - TRUE[name]
    # Noise sigma 2 mm plus a bias uniform in +-1 mm: variance sigma^2 + bias_max^2 / 3.
    variance = 0.002**2 + 0.001**2 / 3
    assert abs(errors.mean()) < 4 * math.sqrt(variance / N)
    # The sample variance of N draws, within the 99.9 % chi-square band (the mix is near normal).
    low, high = chi2.ppf([0.0005, 0.9995], N - 1) * variance / (N - 1)
    assert low < errors.var(ddof=1) < high


def test_the_reported_interval_is_the_estimate_plus_and_minus_k_sigma_and_the_bias():
    seen = looks(n=1)[0]
    for name in TRUE:
        low, high = getattr(seen.dims, name)
        assert high - low == pytest.approx(2 * (4.0 * 0.002 + 0.001))


@pytest.mark.parametrize("name", list(TRUE))
def test_the_true_length_is_inside_its_interval_practically_always(name):
    inside = sum(
        getattr(seen.dims, name)[0] <= TRUE[name] <= getattr(seen.dims, name)[1] for seen in looks()
    )
    # At least 99.99 % is not ruled out by the count: the Wilson upper end reaches it.
    assert wilson_interval(N - inside, N)[0] < 1e-4
    assert inside >= N - 2


def test_no_reported_length_is_negative():
    tiny = dataclasses.replace(BOTTLE, shape=dataclasses.replace(SHAPE, shoulder_height=0.0))
    assert all(seen.dims.shoulder_height[0] >= 0.0 for seen in looks(n=2000, bottle=tiny))


def test_one_look_shares_one_bias_across_its_lengths():
    """Catches a bias drawn per length: with no random noise, every length is off by the same."""
    quiet = dataclasses.replace(NOISE, dim_sigma_m=0.0)
    for seen in looks(n=50, noise=quiet):
        offsets = {round(sum(getattr(seen.dims, name)) / 2 - TRUE[name], 12) for name in TRUE}
        assert len(offsets) == 1
        assert abs(offsets.pop()) <= 0.001
    biases = {round(sum(seen.dims.body_diameter) / 2, 9) for seen in looks(n=50, noise=quiet)}
    assert len(biases) == 50


def test_the_level_of_an_unlabelled_bottle_is_seen_except_for_the_dropouts():
    seen = looks()
    missing = sum(one.visual.level_m is None for one in seen)
    low, high = wilson_interval(missing, N, z=3.29)
    assert low < 0.10 < high
    level = level_for_volume_m(SHAPE, 100.0)
    middles = np.array([sum(one.visual.level_m) / 2 for one in seen if one.visual.level_m])
    assert abs(middles.mean() - level) < 4 * 0.004 / math.sqrt(len(middles))
    assert middles.std() == pytest.approx(0.004, rel=0.05)


def test_a_level_behind_the_label_is_never_seen():
    # The body is 0.15 m tall, so the label hides levels from 0.045 to 0.105 m; 250 mL stands at
    # about 0.077 m in this bottle.
    level = level_for_volume_m(SHAPE, 250.0)
    assert 0.045 < level < 0.105
    assert all(
        one.visual.level_m is None for one in looks(n=2000, bottle=LABELLED, volume_ml=250.0)
    )


def test_a_level_outside_the_label_or_on_a_bottle_without_one_is_seen():
    below = looks(n=2000, bottle=LABELLED, volume_ml=20.0)  # about 6 mm up: under the label
    assert sum(one.visual.level_m is not None for one in below) > 1700
    bare = looks(n=2000, bottle=BOTTLE, volume_ml=250.0)
    assert sum(one.visual.level_m is not None for one in bare) > 1700


def test_the_cap_and_the_base_are_reported_wrongly_at_their_rates():
    seen = looks()
    wrong_cap = sum(one.visual.cap_seen is not True for one in seen)
    wrong_base = sum(one.visual.base_style != "petaloid" for one in seen)
    for wrong, rate in ((wrong_cap, 0.05), (wrong_base, 0.10)):
        low, high = wilson_interval(wrong, N, z=3.29)
        assert low < rate < high
    assert {one.visual.base_style for one in seen} == {"petaloid", "flat"}


def test_a_sound_bottle_is_seen_with_full_confidence_and_a_crushed_one_below_the_minimum():
    minimum = load_policy_config(ROOT / "configs" / "policy.toml").min_confidence
    assert all(one.confidence == 1.0 for one in looks(n=20))
    crushed = looks(n=20, deformed=True)
    assert all(one.confidence == pytest.approx(0.4) for one in crushed)
    assert 0.4 < minimum


def test_a_crushed_bottles_width_interval_is_three_times_as_wide():
    sound, crushed = looks(n=1)[0], looks(n=1, deformed=True)[0]
    width = sound.dims.body_diameter[1] - sound.dims.body_diameter[0]
    assert crushed.dims.body_diameter[1] - crushed.dims.body_diameter[0] == pytest.approx(3 * width)
    assert crushed.dims.total_height == sound.dims.total_height


def test_the_same_generator_state_gives_the_same_look():
    assert looks(n=5, seed=4) == looks(n=5, seed=4)
    assert looks(n=5, seed=4) != looks(n=5, seed=5)
