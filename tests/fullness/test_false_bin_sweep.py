"""How often the rule bins a container it should not: Monte Carlo over bins and sensor quality."""

import dataclasses

import pytest
from support.paths import ROOT

from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.fullness_eval import first_breakdown, run_bin, sweep
from beverage_disposal.core.registry import load_registry

CONFIGS = ROOT / "configs"
FULLNESS = load_fullness_config(CONFIGS / "fullness.toml")
REGISTRY = load_registry(CONFIGS / "containers.toml")
CAN = REGISTRY.lookup("can_12floz")
BOTTLE = REGISTRY.lookup("bottle_500ml")
TRIALS = 100_000
LIGHTEST_DRINK = (
    FULLNESS.rule.min_density_g_per_ml
)  # the worst case for a false bin: least weight per mL
WATER = 1.0


def bins(record, volume_ml, density, tare_kg=None, trials=TRIALS, seed=0, pad=None):
    return run_bin(
        record,
        FULLNESS.rule,
        pad or FULLNESS.pad,
        volume_ml,
        density=density,
        empty_mass_kg=record.lightest_empty_mass if tare_kg is None else tare_kg,
        trials=trials,
        seed=seed,
    )


@pytest.mark.medium
@pytest.mark.parametrize("record", [CAN, BOTTLE], ids=["can", "bottle"])
@pytest.mark.parametrize("volume_ml", [30.0, 31.0, 35.0])
def test_nothing_of_thirty_millilitres_or_more_is_binned_even_of_the_lightest_drink(
    record, volume_ml
):
    result = bins(record, volume_ml, LIGHTEST_DRINK)
    assert result.binned == 0
    # 0 in 100000 trials: the Wilson 95 % upper bound is z^2 / (n + z^2), worked by hand.
    assert result.upper_bound == pytest.approx(3.8416 / (TRIALS + 3.8416), rel=1e-6)
    assert result.upper_bound < 1e-4


@pytest.mark.medium
@pytest.mark.parametrize("record", [CAN, BOTTLE], ids=["can", "bottle"])
def test_an_empty_container_is_binned_almost_every_time_even_at_a_heavier_tare(record):
    # About 7 in a million weighings fail the stability check by chance, so "every time" is 0.9999.
    heavier = record.spec.empty_mass + 0.004  # 4 g over the typical empty weight
    assert bins(record, 0.0, WATER).binned_rate > 0.9999
    assert bins(record, 0.0, WATER, tare_kg=heavier).binned_rate > 0.9999


@pytest.mark.medium
def test_twenty_five_millilitres_of_water_is_binned_almost_always_the_cost_of_caution():
    assert bins(CAN, 25.0, WATER).binned_rate > 0.99


@pytest.mark.medium
def test_twenty_nine_millilitres_is_kept_out_of_the_bin_a_conservative_miss_not_a_failure():
    assert bins(CAN, 29.0, WATER).binned_rate < 0.001


@pytest.mark.medium
def test_a_can_much_heavier_than_its_lightest_empty_looks_full_and_is_set_aside():
    # 30 g over the lightest: the database is what keeps a heavy empty container from being binned.
    assert bins(CAN, 0.0, WATER, tare_kg=CAN.lightest_empty_mass + 0.030).binned_rate < 0.001


@pytest.mark.medium
def test_the_same_seed_gives_the_same_count():
    assert (
        bins(CAN, 26.0, WATER, trials=5000, seed=4).binned
        == bins(CAN, 26.0, WATER, trials=5000, seed=4).binned
    )


@pytest.mark.medium
def test_the_rule_tolerates_twice_the_nominal_noise_and_drift_and_the_sweep_shows_where_it_breaks():
    cells = sweep(
        CAN,
        FULLNESS.rule,
        FULLNESS.pad,
        volume_ml=31.0,
        density=LIGHTEST_DRINK,
        noise_scales=(0.5, 1.0, 2.0, 4.0, 8.0),
        drift_scales=(0.0, 1.0, 2.0),
        trials=TRIALS,
        seed=1,
    )
    by_scale = {(c.noise_scale, c.drift_scale): c for c in cells}
    assert len(cells) == 15
    for noise in (0.5, 1.0, 2.0):
        for drift in (0.0, 1.0, 2.0):
            assert by_scale[(noise, drift)].upper_bound < 0.01, (noise, drift)
    # Worse sensors let more through: the false-bin count never falls as the noise grows.
    counts = [by_scale[(n, 1.0)].binned for n in (0.5, 1.0, 2.0, 4.0, 8.0)]
    assert counts == sorted(counts)
    breakdown = first_breakdown(cells, limit=0.01)
    assert breakdown is not None and breakdown.noise_scale >= 4.0
    assert by_scale[(8.0, 1.0)].upper_bound > 0.01


@pytest.mark.medium
def test_the_sweep_catches_a_rule_with_no_margin():
    # Proof the sweep can fail: without the margin the lightest drink at 31 mL leaks through.
    rule = dataclasses.replace(FULLNESS.rule, margin_g=0.0, min_density_g_per_ml=1.0)
    cells = sweep(
        CAN,
        rule,
        FULLNESS.pad,
        volume_ml=31.0,
        density=LIGHTEST_DRINK,
        noise_scales=(1.0,),
        drift_scales=(1.0,),
        trials=20_000,
        seed=2,
    )
    assert cells[0].upper_bound > 0.01
