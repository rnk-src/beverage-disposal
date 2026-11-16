"""The liquid interval from a weight, and the three zones at their edges."""

import dataclasses

import numpy as np
import pytest
from support.paths import ROOT

from beverage_disposal.core.bottle_family import load_family, sample_bottle
from beverage_disposal.core.bounds import EmptyMassBounds, empty_mass_bounds
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.perception import exact_perception
from beverage_disposal.core.zones import Zone, liquid_bounds_ml, zone_of

RULE = load_fullness_config(ROOT / "configs" / "fullness.toml").rule
FAMILY = load_family(ROOT / "configs" / "bottles.toml")
PET = load_materials(ROOT / "configs" / "materials.toml")["PET"]
BOUNDS = EmptyMassBounds(min_kg=0.010, max_kg=0.030)


def test_the_rule_file_gives_the_drink_densities_and_the_margin():
    assert (RULE.min_density_g_per_ml, RULE.max_density_g_per_ml, RULE.margin_g) == (
        0.95,
        1.05,
        2.0,
    )


def test_the_liquid_interval_by_hand():
    # 50 g on the pad, an empty bottle of 10 to 30 g, a 2 g scale margin.
    v_lo, v_hi = liquid_bounds_ml(0.050, BOUNDS, RULE)
    # Most liquid: the lightest bottle, the scale reading low, the lightest drink.
    assert v_hi == pytest.approx((50 - 10 + 2) / 0.95)
    # Least liquid: the heaviest bottle, the scale reading high, the heaviest drink.
    assert v_lo == pytest.approx((50 - 30 - 2) / 1.05)


def test_the_least_liquid_is_never_negative():
    v_lo, v_hi = liquid_bounds_ml(0.020, BOUNDS, RULE)
    assert v_lo == 0.0 and v_hi == pytest.approx(12 / 0.95)


def test_the_zone_edges_at_exactly_thirty():
    assert zone_of(0.0, 29.999, 30.0) is Zone.SURELY_UNDER
    assert zone_of(0.0, 30.0, 30.0) is Zone.AMBIGUOUS  # it may hold exactly 30 mL
    assert zone_of(29.999, 40.0, 30.0) is Zone.AMBIGUOUS
    assert zone_of(30.0, 40.0, 30.0) is Zone.SURELY_OVER  # it holds 30 mL at the least


def test_no_bottle_of_the_grid_is_put_in_a_zone_its_true_volume_contradicts():
    """Every size, 100 bottles, 0 to 60 mL in half millilitres, three densities, dry and wet."""
    wet_hi = PET.wet_g.hi
    for size in FAMILY.classes:
        for seed in range(100):
            bottle = sample_bottle(FAMILY, size, PET, np.random.default_rng(seed))
            bounds = empty_mass_bounds(exact_perception(bottle, 0.0).dims, PET)
            for volume in np.arange(0.0, 60.5, 0.5):
                for density in (0.95, 1.0, 1.05):
                    for wet in (0.0, wet_hi):
                        true_g = bottle.empty_mass_kg * 1000 + volume * density + wet
                        # A scale off by the whole margin, either way, must not fool the zones.
                        for error in (-RULE.margin_g, 0.0, RULE.margin_g):
                            v_lo, v_hi = liquid_bounds_ml((true_g + error) / 1000, bounds, RULE)
                            assert v_lo - 1e-9 <= volume <= v_hi + 1e-9
                            zone = zone_of(v_lo, v_hi, RULE.threshold_ml)
                            if zone is Zone.SURELY_UNDER:
                                assert volume < 30.0
                            if zone is Zone.SURELY_OVER:
                                assert volume >= 30.0


def test_a_small_empty_bottle_is_surely_under_and_a_half_full_one_surely_over():
    bottle = sample_bottle(FAMILY, "pet_500ml", PET, np.random.default_rng(0))
    bounds = empty_mass_bounds(exact_perception(bottle, 0.0).dims, PET)

    def zone(volume):
        mass = bottle.empty_mass_kg + volume / 1000
        return zone_of(*liquid_bounds_ml(mass, bounds, RULE), RULE.threshold_ml)

    assert zone(0.0) is Zone.SURELY_UNDER
    assert zone(250.0) is Zone.SURELY_OVER


def test_a_heavier_drink_assumption_lowers_only_the_least_liquid():
    syrup = dataclasses.replace(RULE, max_density_g_per_ml=1.3)
    assert liquid_bounds_ml(0.050, BOUNDS, syrup)[0] == pytest.approx(18 / 1.3)
    assert liquid_bounds_ml(0.050, BOUNDS, syrup)[1] == liquid_bounds_ml(0.050, BOUNDS, RULE)[1]
