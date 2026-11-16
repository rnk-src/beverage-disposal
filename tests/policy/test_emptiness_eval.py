"""The decision-level Monte Carlo of the emptiness policy: its counts, seeds and what it catches."""

import dataclasses

import numpy as np
import pytest
from support.paths import ROOT
from test_policy import ALL_SIZES, CONFIG, FAMILY, PET, POLICY, RULE

from beverage_disposal.core.emptiness_eval import (
    Cell,
    draw_case,
    misbin_grid,
    payload_volume_ml,
    summarize_cells,
)
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.pad import PadSensor
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


def case(noise=NOISE, **options):
    rng = np.random.default_rng(0)
    return draw_case(POLICY, PadSensor(PAD, rng), noise, "pet_500ml", 0.0, rng, **options)


def test_a_foreign_object_is_on_the_scale_but_the_level_stays_that_of_the_liquid():
    plain, loaded = case(), case(foreign_g=50.0)
    # The same bottle, drink and look (the same stream); only the scale differs, by the object.
    assert loaded.bottle == plain.bottle and loaded.perception == plain.perception
    assert (loaded.weighing.mass_kg - plain.weighing.mass_kg) * 1000 == pytest.approx(
        50.0, abs=1e-6
    )


def test_a_crushed_bottle_is_seen_as_one_and_needs_the_noise_model():
    assert case().perception.confidence == 1.0
    assert case(deformed=True).perception.confidence == pytest.approx(0.4)
    with pytest.raises(ValueError, match="noise"):
        case(noise=None, deformed=True)


# The acceptance grid: its own bottles, crushed bottles, foreign objects and the summary.


def test_a_stream_gives_the_grid_bottles_of_its_own():
    plain = noisy(volumes_ml=(0.0, 35.0))
    assert noisy(volumes_ml=(0.0, 35.0), stream=(3, 2)) != plain
    assert noisy(volumes_ml=(0.0, 35.0), stream=(3, 2)) == noisy(
        volumes_ml=(0.0, 35.0), stream=(3, 2)
    )
    assert noisy(volumes_ml=(0.0, 35.0), stream=()) == plain


def test_every_crushed_bottle_of_the_grid_is_vetoed_and_the_cell_says_it_was_crushed():
    (cell,) = noisy(deformed=True, trials=100)
    assert (cell.binned, cell.deformed) == (0, True)
    assert cell.decided_by == {"veto:low_confidence": 100}
    assert noisy(trials=100)[0].deformed is False


def test_a_foreign_object_of_the_grid_is_on_the_scale_and_named_in_the_cell():
    (cell,) = noisy(foreign_g=100.0, trials=100)
    assert (cell.binned, cell.foreign_g) == (0, 100.0)
    # A sound empty 500 mL bottle is nearly always binned, so the object is what stopped these.
    (sound,) = noisy(trials=100)
    assert sound.binned > 90 and sound.foreign_g == 0.0


def with_model(score, tau):
    from test_full_policy import bundled

    return bundled(score, tau=tau)


def test_a_cell_counts_the_bins_the_classifier_made_in_the_ambiguous_zone():
    options = dict(sizes=("pet_2000ml",), volumes_ml=(20.0,), trials=100, seed=0, noise=NOISE)
    (eager,) = misbin_grid(with_model(1.0, 0.0), FAMILY, PET, PAD, **options)
    # Not every ambiguous bottle: one the noisy camera fits to two sizes is left to the rule.
    assert eager.zones == {"ambiguous": 100}
    assert eager.binned == eager.overturned == eager.decided_by["classifier"] > 50
    (alone,) = misbin_grid(POLICY, FAMILY, PET, PAD, **options)
    assert alone.overturned == 0 and alone.zones == eager.zones
    # A bin the rule made is not an overturn.
    (ruled,) = noisy(trials=100)
    assert ruled.binned > 90 and ruled.overturned == 0


def test_a_classifiers_veto_is_not_counted_as_an_overturn():
    (cell,) = misbin_grid(
        with_model(0.0, 1.0), FAMILY, PET, PAD, sizes=("pet_500ml",), volumes_ml=(0.0,), trials=50,
        seed=0, noise=NOISE,
    )  # fmt: skip
    assert cell.decided_by["classifier"] > 40 and (cell.binned, cell.overturned) == (0, 0)


def test_the_payload_volume_is_the_nominal_size_or_what_the_arm_may_lift():
    assert payload_volume_ml(FAMILY, "pet_330ml", payload_kg=0.6) == 330.0
    assert payload_volume_ml(FAMILY, "pet_2000ml", payload_kg=0.6) == 600.0
    assert payload_volume_ml(FAMILY, "pet_2000ml", payload_kg=0.5) == 500.0


def a_cell(size, volume_ml, binned, trials=100, **more):
    return Cell(size, volume_ml, trials, binned, wilson_upper(binned, trials), {}, {}, **more)


def test_the_summary_counts_misbins_empties_overturns_and_vetoes_by_hand():
    cells = [
        a_cell("a", 0.0, 90, overturned=20),
        a_cell("a", 29.0, 50),  # under the threshold: not a misbin
        a_cell("a", 30.0, 0),
        a_cell("a", 35.0, 2, overturned=2),
        a_cell("b", 0.0, 10),
        a_cell("b", 31.0, 1, trials=50),
        a_cell("a", 0.0, 3, deformed=True),
        a_cell("b", 0.0, 4, foreign_g=50.0),
        a_cell("b", 0.0, 5, foreign_g=100.0),
    ]
    cells[2] = dataclasses.replace(cells[2], decided_by={"rule": 60, "veto:unsettled": 40})
    cells[4] = dataclasses.replace(cells[4], decided_by={"veto:unsettled": 5, "veto:not_pet": 1})
    summary = summarize_cells(cells, threshold_ml=30.0)
    assert summary["binned_at_or_over_threshold"] == 3
    assert summary["trials_at_or_over_threshold"] == 250
    assert summary["worst_upper_95_at_or_over_threshold"] == wilson_upper(1, 50)
    assert summary["empties_binned"] == {"a": [90, 100], "b": [10, 100]}
    assert summary["overturned"] == 22
    assert summary["vetoes"] == {"veto:unsettled": 45, "veto:not_pet": 1}
    assert summary["crushed_binned"] == {"a": [3, 100]}
    assert summary["foreign_binned"] == {"b": {"50": [4, 100], "100": [5, 100]}}


def test_a_crushed_or_foreign_cell_is_not_counted_among_the_sound_bottles():
    cells = [a_cell("a", 35.0, 7, deformed=True), a_cell("a", 0.0, 9, foreign_g=50.0)]
    summary = summarize_cells(cells, threshold_ml=30.0)
    assert summary["binned_at_or_over_threshold"] == 0 and summary["empties_binned"] == {}
