"""The simulated wall and liquid probes: readings are hard bounds, randomness has its own stream."""

import dataclasses
import math

import numpy as np
import pytest
from support.paths import ROOT

from beverage_disposal.core.bottles import Bottle, BottleShape
from beverage_disposal.core.probes import (
    NO_PROBES,
    LiquidProbeModel,
    ProbeModels,
    ProbeReadings,
    WallProbeModel,
    load_probe_models,
    probe,
)

PATH = ROOT / "configs" / "probes.toml"
MODELS = load_probe_models(PATH)
N = 20_000
T_SIDE_MM = 0.2
SHAPE = BottleShape(
    body_diameter=0.065,
    body_height=0.15,
    shoulder_height=0.04,
    neck_diameter=0.026,
    neck_height=0.02,
    base_style="petaloid",
    t_side=T_SIDE_MM / 1000,
    t_base=0.0005,
    finish_mass=0.002,
    cap_mass=0.0015,
    label_mass=0.0,
)
BOTTLE = Bottle(SHAPE, 1.38, "pet_500ml")
EXACT_WALL = WallProbeModel(rel_sigma=0.0, interval_k=4.0, nonuniformity=0.0, dropout=0.0)
EXACT_LIQUID = LiquidProbeModel(sigma_ml=0.0, rel_sigma=0.0, interval_k=4.0, dropout=0.0)
EXACT = ProbeModels(wall=EXACT_WALL, liquid=EXACT_LIQUID)
# The configured errors with no dropout, so every try gives a reading to measure.
ALWAYS = ProbeModels(
    wall=dataclasses.replace(MODELS.wall, dropout=0.0),
    liquid=dataclasses.replace(MODELS.liquid, dropout=0.0),
)


class Scripted:
    """A generator that hands back chosen numbers, so a reading can be worked out by hand."""

    def __init__(self, uniform, normal):
        self._uniform = list(uniform)
        self._normal = list(normal)

    def random(self):
        return self._uniform.pop(0)

    def normal(self):
        return self._normal.pop(0)


def readings(n=N, models=ALWAYS, volume_ml=50.0, seed=0, **options):
    rng = np.random.default_rng(seed)
    return [probe(BOTTLE, volume_ml, models, rng, **options) for _ in range(n)]


def test_the_probe_file_holds_the_assumed_figures():
    assert MODELS.wall == WallProbeModel(0.05, 4.0, 0.10, 0.05)
    assert MODELS.liquid == LiquidProbeModel(2.0, 0.10, 4.0, 0.05)
    with pytest.raises(ValueError):
        load_probe_models(ROOT / "configs" / "policy.toml")


def test_a_probe_file_missing_a_table_or_with_an_extra_key_is_refused(tmp_path):
    text = PATH.read_text()
    wall_only = tmp_path / "wall.toml"
    wall_only.write_text(text.split("[liquid]")[0])
    with pytest.raises(ValueError):
        load_probe_models(wall_only)
    extra = tmp_path / "extra.toml"
    extra.write_text("surprise = 1\n" + text)
    with pytest.raises(ValueError):
        load_probe_models(extra)


def test_no_fitted_probe_reports_nothing():
    assert probe(BOTTLE, 50.0, ProbeModels(), np.random.default_rng(0)) == NO_PROBES
    assert NO_PROBES == ProbeReadings(None, None)


def test_every_draw_happens_whether_or_not_a_probe_is_fitted():
    fitted, bare = np.random.default_rng(3), np.random.default_rng(3)
    probe(BOTTLE, 50.0, MODELS, fitted)
    probe(BOTTLE, 50.0, ProbeModels(), bare)
    by_hand = np.random.default_rng(3)
    by_hand.random(), by_hand.normal(), by_hand.random(), by_hand.normal(), by_hand.normal()
    assert fitted.random() == bare.random() == by_hand.random()


def test_the_wall_estimate_is_right_on_average_and_spread_as_configured():
    # The estimate is the reading's low end times (1 + half-width), see the next test.
    half = MODELS.wall.half_width
    ratios = np.array([r.side_wall_mm[0] * (1 + half) / T_SIDE_MM for r in readings()])
    assert abs(ratios.mean() - 1.0) < 4 * 0.05 / math.sqrt(N)
    assert abs(ratios.std() - 0.05) < 0.05 * 0.05  # the sample sigma of 20 000 draws is within 5 %


def test_the_wall_reading_is_the_estimate_over_one_plus_and_minus_the_half_width():
    model = WallProbeModel(rel_sigma=0.05, interval_k=4.0, nonuniformity=0.10, dropout=0.0)
    # half width 4 * 0.05 + 0.10 = 0.30; estimate 0.2 * (1 + 0.05 * 2) = 0.22 mm.
    rng = Scripted(uniform=[0.5, 0.5], normal=[2.0, 0.0, 0.0])
    seen = probe(BOTTLE, 0.0, ProbeModels(wall=model), rng)
    assert seen.side_wall_mm == pytest.approx((0.22 / 1.3, 0.22 / 0.7))
    assert seen.side_wall_mm[0] < T_SIDE_MM < seen.side_wall_mm[1]


def test_the_true_wall_is_inside_its_reading_practically_always():
    outside = [r for r in readings() if not r.side_wall_mm[0] <= T_SIDE_MM <= r.side_wall_mm[1]]
    assert outside == []  # six sigma either way


def test_a_cap_makes_no_difference_to_the_wall_reading():
    uncapped = Bottle(dataclasses.replace(SHAPE, cap_mass=0.0), 1.38, "pet_500ml")
    with_cap = probe(BOTTLE, 50.0, MODELS, np.random.default_rng(5))
    without = probe(uncapped, 50.0, MODELS, np.random.default_rng(5))
    assert with_cap.side_wall_mm == without.side_wall_mm


def test_contents_the_probe_cannot_see_read_as_empty():
    seen = probe(BOTTLE, 200.0, EXACT, np.random.default_rng(0), liquid_gain=0.0)
    assert seen.liquid_ml == (0.0, 0.0)


def test_the_wall_and_the_liquid_drop_out_at_their_rates():
    wall = dataclasses.replace(EXACT_WALL, dropout=0.30)
    liquid = dataclasses.replace(EXACT_LIQUID, dropout=0.10)
    seen = readings(models=ProbeModels(wall=wall, liquid=liquid))
    for rate, missing in (
        (0.30, sum(r.side_wall_mm is None for r in seen)),
        (0.10, sum(r.liquid_ml is None for r in seen)),
    ):
        assert abs(missing / N - rate) < 4 * math.sqrt(rate * (1 - rate) / N)


def test_the_liquid_reading_by_hand():
    exact = probe(BOTTLE, 10.0, EXACT, np.random.default_rng(0))
    assert exact.liquid_ml == (10.0, 10.0)
    model = LiquidProbeModel(sigma_ml=2.0, rel_sigma=0.1, interval_k=4.0, dropout=0.0)
    rng = Scripted(uniform=[0.5, 0.5], normal=[0.0, 1.0, -1.0])
    seen = probe(BOTTLE, 50.0, ProbeModels(liquid=model), rng)
    # estimate 50 * 1.1 - 2 = 53; relative slack 0.4, additive slack 8 mL.
    assert seen.liquid_ml == pytest.approx((45.0 / 1.4, 61.0 / 0.6))


def test_the_true_liquid_is_inside_its_reading_practically_always():
    outside = sum(not r.liquid_ml[0] <= 50.0 <= r.liquid_ml[1] for r in readings())
    # Two errors, each outside 4 sigma about 6e-5 of the time: about 2.5 of 20 000 expected.
    assert outside / N < 5e-4


def test_a_low_reading_probe_puts_the_truth_above_its_reading():
    seen = probe(BOTTLE, 50.0, EXACT, np.random.default_rng(0), liquid_gain=0.5)
    assert seen.liquid_ml == (25.0, 25.0)
    assert seen.liquid_ml[1] < 50.0


def test_no_liquid_reading_is_negative():
    for seen in readings(volume_ml=0.0, n=2000):
        assert 0.0 <= seen.liquid_ml[0] <= seen.liquid_ml[1]


def test_fitting_the_wall_probe_does_not_change_the_liquid_reading():
    both = probe(BOTTLE, 50.0, MODELS, np.random.default_rng(9))
    liquid_only = probe(
        BOTTLE, 50.0, dataclasses.replace(MODELS, wall=None), np.random.default_rng(9)
    )
    assert both.liquid_ml == liquid_only.liquid_ml
    assert liquid_only.side_wall_mm is None


def test_the_same_generator_state_gives_the_same_readings():
    assert readings(n=50, seed=4) == readings(n=50, seed=4)
    assert readings(n=50, seed=4) != readings(n=50, seed=5)


def test_a_model_with_a_half_width_of_one_or_more_is_refused():
    with pytest.raises(ValueError, match="half"):
        WallProbeModel(rel_sigma=0.2, interval_k=4.0, nonuniformity=0.2, dropout=0.0)
    WallProbeModel(rel_sigma=0.2, interval_k=4.0, nonuniformity=0.19, dropout=0.0)


@pytest.mark.parametrize(
    "build",
    [
        lambda: WallProbeModel(-0.1, 4.0, 0.1, 0.0),
        lambda: WallProbeModel(0.1, 4.0, -0.1, 0.0),
        lambda: WallProbeModel(0.1, 4.0, 0.1, 1.5),
        lambda: WallProbeModel(0.1, 4.0, 0.1, -0.1),
        lambda: LiquidProbeModel(-1.0, 0.1, 4.0, 0.0),
        lambda: LiquidProbeModel(1.0, -0.1, 4.0, 0.0),
        lambda: LiquidProbeModel(1.0, 0.25, 4.0, 0.0),
        lambda: LiquidProbeModel(1.0, 0.1, 4.0, 1.5),
    ],
)
def test_a_model_with_impossible_figures_is_refused(build):
    with pytest.raises(ValueError):
        build()


def test_a_reading_with_its_ends_swapped_is_refused():
    with pytest.raises(ValueError):
        ProbeReadings(side_wall_mm=(0.3, 0.2))
    with pytest.raises(ValueError):
        ProbeReadings(liquid_ml=(5.0, 4.0))
    ProbeReadings(liquid_ml=(4.0, 4.0))
