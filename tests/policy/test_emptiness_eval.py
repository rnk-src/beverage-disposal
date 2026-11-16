"""The decision-level Monte Carlo of the emptiness policy: its counts, seeds and what it catches."""

import dataclasses

import pytest
from support.paths import ROOT
from test_policy import ALL_SIZES, CONFIG, FAMILY, PET, POLICY, RULE

from beverage_disposal.core.emptiness_eval import misbin_grid
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.perception import load_perception_noise
from beverage_disposal.core.policy import EmptinessPolicy
from beverage_disposal.core.statistics import wilson_upper

PAD = load_fullness_config(ROOT / "configs" / "fullness.toml").pad
SMALL = ("pet_250ml", "pet_500ml")


def grid(policy=POLICY, sizes=SMALL, volumes=(0.0, 35.0), trials=40, seed=0):
    return misbin_grid(
        policy, FAMILY, PET, PAD, sizes=sizes, volumes_ml=volumes, trials=trials, seed=seed
    )


def test_there_is_one_cell_per_size_and_volume_with_every_trial_accounted_for():
    cells = grid()
    assert [(c.size, c.volume_ml) for c in cells] == [
        ("pet_250ml", 0.0),
        ("pet_250ml", 35.0),
        ("pet_500ml", 0.0),
        ("pet_500ml", 35.0),
    ]
    for cell in cells:
        assert cell.trials == 40 == sum(cell.decided_by.values())
        assert sum(cell.zones.values()) <= 40
        assert cell.upper_95 == wilson_upper(cell.binned, 40)


def test_small_empty_bottles_are_binned_and_bottles_of_35_ml_never():
    cells = {(c.size, c.volume_ml): c for c in grid()}
    assert cells[("pet_500ml", 0.0)].binned > 30
    assert cells[("pet_500ml", 35.0)].binned == 0 == cells[("pet_250ml", 35.0)].binned


def test_the_same_seed_gives_the_same_cells_and_another_seed_other_bottles():
    big = dict(sizes=("pet_2000ml",), volumes=(10.0,), trials=60)
    assert grid(**big, seed=1) == grid(**big, seed=1)
    assert grid(**big, seed=1) != grid(**big, seed=2)


def test_a_rule_with_the_wrong_threshold_shows_up_as_binned_bottles_of_35_ml():
    """Catches a grid that cannot see a misbin: this policy bins anything under 1000 mL."""
    loose = dataclasses.replace(POLICY, rule=dataclasses.replace(RULE, threshold_ml=1000.0))
    cell = grid(loose, sizes=("pet_500ml",), volumes=(35.0,))[0]
    assert cell.binned == 40 and cell.zones == {"surely_under": 40}


def test_a_bottle_holding_exactly_30_ml_is_never_proven_either_way():
    # The weight cannot prove under 30 mL (it is not) nor 30 mL or more (the bottle may be heavy).
    cells = grid(sizes=ALL_SIZES, volumes=(30.0, 31.0), trials=30)
    assert all(cell.binned == 0 and cell.zones == {"ambiguous": 30} for cell in cells)


def mean_weight_g(rule=RULE, material=PET):
    """The mean pad reading over one cell of 40 bottles of 500 mL holding 100 mL."""
    seen = []

    class Spy(EmptinessPolicy):
        def decide(self, perception, weighing):
            seen.append(weighing.mass_kg)
            return super().decide(perception, weighing)

    misbin_grid(
        Spy(rule, material, FAMILY, CONFIG, sizes=ALL_SIZES),
        FAMILY,
        material,
        PAD,
        sizes=("pet_500ml",),
        volumes_ml=(100.0,),
        trials=40,
        seed=0,
    )
    return 1000 * sum(seen) / len(seen)


def test_the_drink_is_weighed_at_its_density():
    # The same bottles and the same pad noise; only the drink changes, by 0.10 g/mL on 100 mL.
    light = dataclasses.replace(RULE, min_density_g_per_ml=0.95, max_density_g_per_ml=0.95)
    heavy = dataclasses.replace(RULE, min_density_g_per_ml=1.05, max_density_g_per_ml=1.05)
    assert mean_weight_g(heavy) - mean_weight_g(light) == pytest.approx(10.0, abs=0.05)


def test_water_on_the_outside_is_on_the_scale():
    def wet(grams):
        return dataclasses.replace(PET, wet_g=dataclasses.replace(PET.wet_g, lo=grams, hi=grams))

    assert mean_weight_g(material=wet(5.0)) - mean_weight_g(material=wet(0.0)) == pytest.approx(
        5.0, abs=0.05
    )


def test_a_policy_that_names_no_size_vetoes_every_bottle():
    nothing = EmptinessPolicy(RULE, PET, FAMILY, CONFIG, sizes=())
    cell = grid(nothing, sizes=("pet_500ml",), volumes=(0.0,))[0]
    assert cell.binned == 0 and cell.decided_by == {"veto:unsupported_size": 40}


@pytest.mark.medium
def test_no_500_ml_bottle_holding_30_ml_is_binned_in_400_trials():
    cell = grid(sizes=("pet_500ml",), volumes=(30.0,), trials=400)[0]
    assert cell.binned == 0 and cell.upper_95 < 0.01


NOISE = load_perception_noise(ROOT / "configs" / "perception.toml")


def noisy(**options):
    defaults = dict(sizes=("pet_500ml",), volumes_ml=(0.0,), trials=200, seed=0)
    return misbin_grid(POLICY, FAMILY, PET, PAD, noise=NOISE, **{**defaults, **options})


def test_a_noisy_camera_bins_fewer_empties_than_an_exact_one():
    # At 10 mL about half the 500 mL bottles are provably under 30 mL: wider intervals cost some.
    exact = grid(sizes=("pet_500ml",), volumes=(10.0,), trials=400)[0]
    seen = noisy(volumes_ml=(10.0,), trials=400)[0]
    assert 0 < seen.binned < exact.binned - 10


def test_a_noisy_camera_still_bins_no_bottle_of_30_ml_or_more():
    cells = noisy(sizes=ALL_SIZES, volumes_ml=(30.0, 31.0, 35.0), trials=100)
    assert all(cell.binned == 0 for cell in cells)


def test_a_wider_reported_interval_never_bins_more():
    """Catches intervals that do not reach the bounds: a vaguer camera cannot be more sure."""
    vague = dataclasses.replace(NOISE, interval_k=8.0)
    wide = misbin_grid(
        POLICY,
        FAMILY,
        PET,
        PAD,
        noise=vague,
        sizes=("pet_500ml",),
        volumes_ml=(0.0,),
        trials=200,
        seed=0,
    )[0]
    assert wide.binned < noisy()[0].binned


@pytest.mark.medium
def test_no_500_ml_bottle_holding_30_ml_is_binned_in_400_trials_with_a_noisy_camera():
    cell = noisy(volumes_ml=(30.0,), trials=400)[0]
    assert cell.binned == 0 and cell.upper_95 < 0.01
