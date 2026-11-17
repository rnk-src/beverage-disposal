"""The rule-only emptiness policy: each veto alone, their order, and the three zones."""

import dataclasses
import math
import tomllib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support.paths import ROOT

from beverage_disposal.core.bottle_family import (
    GateRecord,
    classes_matching,
    load_family,
    sample_bottle,
)
from beverage_disposal.core.bounds import empty_mass_bounds
from beverage_disposal.core.features import level_volume_bounds_ml
from beverage_disposal.core.fullness import BIN, SET_ASIDE
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.pad import Weighing
from beverage_disposal.core.perception import exact_perception
from beverage_disposal.core.policy import EmptinessPolicy, load_policy_config
from beverage_disposal.core.probes import NO_PROBES, ProbeReadings
from beverage_disposal.core.zones import Zone, intersect_ml, liquid_bounds_ml

CONFIGS = ROOT / "configs"
RULE = load_fullness_config(CONFIGS / "fullness.toml").rule
FAMILY = load_family(CONFIGS / "bottles.toml")
PET = load_materials(CONFIGS / "materials.toml")["PET"]
CONFIG = load_policy_config(CONFIGS / "policy.toml")
ALL_SIZES = tuple(FAMILY.classes)
POLICY = EmptinessPolicy(RULE, PET, FAMILY, CONFIG, sizes=ALL_SIZES)
BOTTLE = sample_bottle(FAMILY, "pet_500ml", PET, np.random.default_rng(0))
BIG = sample_bottle(FAMILY, "pet_2000ml", PET, np.random.default_rng(0))
# Neighbouring sizes share widths and heights, so the 500 mL bottle fits more than its own class.
BOTTLE_MATCHES = classes_matching(FAMILY, exact_perception(BOTTLE, 0.0).dims)
assert "pet_500ml" in BOTTLE_MATCHES and len(BOTTLE_MATCHES) > 1


def weighed(bottle, volume_ml, **changes):
    mass = bottle.empty_mass_kg + volume_ml / 1000
    return Weighing(**{"mass_kg": mass, "stable": True, **changes})


def decide(bottle=BOTTLE, volume_ml=0.0, policy=POLICY, weighing=None, **seen):
    perception = dataclasses.replace(exact_perception(bottle, volume_ml), **seen)
    return policy.decide(perception, weighing or weighed(bottle, volume_ml))


def test_the_policy_file_holds_the_decided_figures():
    raw = tomllib.loads((CONFIGS / "policy.toml").read_text())
    assert (CONFIG.alpha, CONFIG.delta) == (0.005, 0.05) == (raw["alpha"], raw["delta"])
    assert CONFIG.volume_bins == (30.0, 31.0, 35.0, 50.0)
    # Chosen 2026-11-16: no bin rests on the camera's level reading.
    assert CONFIG.allow_overturn is False and CONFIG.min_confidence == 0.8


def test_an_empty_bottle_whose_weight_proves_it_is_binned_by_the_rule():
    verdict = decide(volume_ml=0.0)
    assert (verdict.action, verdict.decided_by, verdict.zone) == (BIN, "rule", "surely_under")


def test_a_bottle_whose_weight_proves_30_ml_or_more_is_set_aside_by_the_rule():
    verdict = decide(volume_ml=250.0)
    assert (verdict.action, verdict.decided_by, verdict.zone) == (SET_ASIDE, "rule", "surely_over")


def test_a_bottle_in_the_ambiguous_zone_is_set_aside_without_a_classifier():
    # A big bottle's empty weight is too uncertain: 20 mL in it could be 20 mL or far more.
    verdict = decide(BIG, volume_ml=20.0)
    assert (verdict.action, verdict.decided_by, verdict.zone) == (SET_ASIDE, "rule", "ambiguous")


def test_every_zone_verdict_carries_the_liquid_bounds_it_was_made_from():
    verdict = decide(volume_ml=10.0)
    bounds = empty_mass_bounds(exact_perception(BOTTLE, 10.0).dims, PET)
    assert verdict.bounds_ml == pytest.approx(
        liquid_bounds_ml(weighed(BOTTLE, 10.0).mass_kg, bounds, RULE)
    )
    assert verdict.net_g is None and verdict.score is None


@pytest.mark.parametrize(
    ("code", "kwargs"),
    [
        ("not_pet", {"material": "glass"}),
        ("unsettled", {"weighing": weighed(BOTTLE, 0.0, stable=False)}),
        ("low_confidence", {"confidence": 0.79}),
        ("lighter_than_possible", {"weighing": Weighing(mass_kg=0.0005, stable=True)}),
    ],
)
def test_each_veto_alone_sets_an_otherwise_empty_bottle_aside_under_its_own_code(code, kwargs):
    verdict = decide(volume_ml=0.0, **kwargs)
    assert verdict.action == SET_ASIDE and verdict.decided_by == f"veto:{code}"


def test_a_confidence_exactly_at_the_minimum_is_acted_on():
    assert decide(confidence=0.8).action == BIN


def test_a_size_that_matches_no_class_is_vetoed():
    odd = dataclasses.replace(exact_perception(BOTTLE, 0.0).dims, body_diameter=(0.2, 0.2))
    verdict = decide(dims=odd)
    assert verdict.action == SET_ASIDE and verdict.decided_by == "veto:unsupported_size"


def test_a_size_outside_the_policys_sizes_is_vetoed_and_one_inside_is_not():
    small_only = dataclasses.replace(POLICY, sizes=("pet_250ml",))
    assert decide(policy=small_only).decided_by == "veto:unsupported_size"
    assert decide(policy=dataclasses.replace(POLICY, sizes=BOTTLE_MATCHES)).action == BIN


def test_a_bottle_that_could_also_be_an_unsupported_size_is_vetoed():
    # 56 mm wide and 150 mm tall fits both 250 and 330 mL; only one of them is supported.
    both = dataclasses.replace(
        exact_perception(BOTTLE, 0.0).dims, body_diameter=(0.056, 0.056), total_height=(0.15, 0.15)
    )
    one = dataclasses.replace(POLICY, sizes=("pet_250ml",))
    two = dataclasses.replace(POLICY, sizes=("pet_250ml", "pet_330ml"))
    light = Weighing(mass_kg=0.012, stable=True)
    assert decide(policy=one, dims=both, weighing=light).decided_by == "veto:unsupported_size"
    assert decide(policy=two, dims=both, weighing=light).decided_by == "rule"


def test_without_named_sizes_the_policy_acts_only_on_sizes_that_passed_the_gate():
    default = EmptinessPolicy(RULE, PET, FAMILY, CONFIG)
    assert decide(policy=default).decided_by == "veto:unsupported_size"  # no size has passed
    passed = {
        name: dataclasses.replace(FAMILY.classes[name], gate=GateRecord("passed", "2026-11-16", ""))
        for name in BOTTLE_MATCHES
    }
    family = dataclasses.replace(FAMILY, classes={**FAMILY.classes, **passed})
    assert decide(policy=EmptinessPolicy(RULE, PET, family, CONFIG)).action == BIN


def test_the_vetoes_are_checked_in_the_stated_order():
    everything_wrong = {
        "material": "glass",
        "confidence": 0.1,
        "weighing": weighed(BOTTLE, 0.0, stable=False),
        "dims": dataclasses.replace(exact_perception(BOTTLE, 0.0).dims, body_diameter=(0.2, 0.2)),
    }
    order = ["not_pet", "unsettled", "low_confidence", "unsupported_size"]
    fixes = {
        "not_pet": {"material": "PET"},
        "unsettled": {"weighing": weighed(BOTTLE, 0.0)},
        "low_confidence": {"confidence": 1.0},
    }
    wrong = dict(everything_wrong)
    for code in order:
        assert decide(**wrong).decided_by == f"veto:{code}"
        wrong.update(fixes.get(code, {}))


def test_a_veto_stands_even_when_the_weight_says_surely_under():
    verdict = decide(volume_ml=0.0, material="glass")
    assert verdict.action == SET_ASIDE and verdict.zone is None


SMALL = sample_bottle(FAMILY, "pet_250ml", PET, np.random.default_rng(0))


def test_a_weight_that_proves_far_more_liquid_than_the_level_shows_is_vetoed():
    """Catches a solid object in an empty bottle passing for liquid: the camera sees none."""
    # The level of an empty bottle, the weight of one with 150 g inside.
    heavy = weighed(SMALL, 150.0)
    verdict = POLICY.decide(exact_perception(SMALL, 0.0), heavy)
    assert verdict.action == SET_ASIDE
    assert verdict.decided_by == "veto:inconsistent_weight_and_level"


def test_a_weight_and_a_level_that_agree_are_left_to_the_zones():
    verdict = POLICY.decide(exact_perception(SMALL, 150.0), weighed(SMALL, 150.0))
    assert (verdict.decided_by, verdict.zone) == ("rule", "surely_over")


def test_an_unseen_level_cannot_contradict_the_weight():
    unseen = exact_perception(SMALL, 0.0)
    unseen = dataclasses.replace(unseen, visual=dataclasses.replace(unseen.visual, level_m=None))
    assert POLICY.decide(unseen, weighed(SMALL, 150.0)).decided_by == "rule"


def test_the_slack_is_how_much_more_the_weight_may_show_before_the_veto():
    assert CONFIG.level_slack_ml == 20.0
    heavy = weighed(SMALL, 150.0)
    assessment = POLICY.assess(exact_perception(SMALL, 0.0), heavy)
    excess = assessment.v_bounds[0] - assessment.level_ml[1]
    assert excess > 20.0
    # A policy that tolerates just more than this excess does not veto, one just under does.
    tolerant = dataclasses.replace(
        POLICY, config=dataclasses.replace(CONFIG, level_slack_ml=excess + 0.01)
    )
    strict = dataclasses.replace(
        POLICY, config=dataclasses.replace(CONFIG, level_slack_ml=excess - 0.01)
    )
    assert tolerant.decide(exact_perception(SMALL, 0.0), heavy).decided_by == "rule"
    assert strict.decide(exact_perception(SMALL, 0.0), heavy).decided_by.startswith("veto:")


def test_the_assessment_holds_what_the_verdict_was_made_from():
    perception, weighing = exact_perception(BOTTLE, 10.0), weighed(BOTTLE, 10.0)
    assessment = POLICY.assess(perception, weighing)
    verdict = POLICY.decide(perception, weighing)
    assert assessment.veto is None and assessment.zone.value == verdict.zone
    assert assessment.v_bounds == verdict.bounds_ml
    assert assessment.bounds == empty_mass_bounds(perception.dims, PET)
    vetoed = POLICY.assess(dataclasses.replace(perception, material="glass"), weighing)
    assert vetoed.veto.decided_by == "veto:not_pet" and vetoed.zone is None


def test_a_level_interval_that_may_reach_the_weights_liquid_is_not_a_contradiction():
    # The camera says the level is somewhere between the floor and well up the body: the 150 g
    # may be liquid after all, so only the high end of the level counts.
    perception = exact_perception(SMALL, 0.0)
    vague = dataclasses.replace(
        perception, visual=dataclasses.replace(perception.visual, level_m=(0.0, 0.10))
    )
    assert POLICY.decide(vague, weighed(SMALL, 150.0)).decided_by == "rule"


# --- probes (2B.14) ---------------------------------------------------------------------------

WALL_HALF = 0.10


def true_wall_reading(bottle, half=WALL_HALF):
    """A wall reading that holds the bottle's true wall, in mm."""
    t = bottle.shape.t_side * 1000
    return (t / (1 + half), t / (1 - half))


def reach_of(reach_ml):
    return dataclasses.replace(POLICY, config=dataclasses.replace(CONFIG, probe_reach_ml=reach_ml))


UNLIMITED = reach_of(math.inf)  # the fusing tests below are about the fusion, not the reach


def test_without_probes_the_verdict_is_what_it_was():
    rng = np.random.default_rng(11)
    for size in ("pet_250ml", "pet_500ml", "pet_1000ml", "pet_2000ml"):
        for volume in (0.0, 10.0, 29.0, 31.0, 80.0):
            bottle = sample_bottle(FAMILY, size, PET, rng)
            perception, weighing = exact_perception(bottle, volume), weighed(bottle, volume)
            assert POLICY.decide(perception, weighing) == POLICY.decide(
                perception, weighing, NO_PROBES
            )
            assert POLICY.assess(perception, weighing) == POLICY.assess(
                perception, weighing, NO_PROBES
            )


def test_a_wall_reading_bins_an_empty_2000_ml_bottle_the_weight_alone_cannot():
    alone = decide(BIG, 0.0)
    assert (alone.action, alone.zone) == (SET_ASIDE, "ambiguous")
    with_wall = POLICY.decide(
        exact_perception(BIG, 0.0),
        weighed(BIG, 0.0),
        ProbeReadings(side_wall_mm=true_wall_reading(BIG)),
    )
    # Worked by hand in the study's notes: the weight's liquid interval falls from 0 to 43.8 mL
    # to 0 to 25.1 mL, under the 30 mL threshold.
    assert (with_wall.action, with_wall.zone, with_wall.decided_by) == (BIN, "surely_under", "rule")
    assert with_wall.bounds_ml[1] < 30.0


def test_a_wall_reading_outside_the_materials_range_is_a_veto():
    verdict = POLICY.decide(
        exact_perception(BIG, 0.0), weighed(BIG, 0.0), ProbeReadings(side_wall_mm=(0.50, 0.60))
    )
    assert (verdict.action, verdict.decided_by) == (SET_ASIDE, "veto:implausible_wall")


def test_the_camera_seeing_more_than_the_liquid_probe_is_a_veto():
    perception, weighing = exact_perception(BOTTLE, 100.0), weighed(BOTTLE, 100.0)
    level_low = level_volume_bounds_ml(perception, PET)[0]
    slack = CONFIG.level_slack_ml
    just_outside = ProbeReadings(liquid_ml=(0.0, level_low - slack - 0.01))
    just_inside = ProbeReadings(liquid_ml=(0.0, level_low - slack + 0.01))
    seen = POLICY.decide(perception, weighing, just_outside)
    assert (seen.action, seen.decided_by) == (SET_ASIDE, "veto:inconsistent_level_and_probe")
    near_miss = POLICY.decide(perception, weighing, just_inside)
    assert near_miss.decided_by != "veto:inconsistent_level_and_probe"
    assert near_miss.action == SET_ASIDE  # the probe still contradicts the weight


def test_an_unseen_level_leaves_the_liquid_probe_alone():
    perception, weighing = exact_perception(BOTTLE, 100.0), weighed(BOTTLE, 100.0)
    blind = dataclasses.replace(
        perception, visual=dataclasses.replace(perception.visual, level_m=None)
    )
    reading = ProbeReadings(liquid_ml=(0.0, 10.0))
    assert UNLIMITED.decide(perception, weighing, reading).decided_by == (
        "veto:inconsistent_level_and_probe"
    )
    assert (
        UNLIMITED.decide(blind, weighing, reading).decided_by
        == "veto:inconsistent_weight_and_probe"
    )


def test_a_liquid_reading_is_dropped_when_the_weight_allows_more_than_the_reach():
    perception, weighing = exact_perception(BIG, 10.0), weighed(BIG, 10.0)
    reading = ProbeReadings(liquid_ml=(5.0, 15.0))
    weight_high = POLICY.assess(perception, weighing).v_bounds[1]
    assert weight_high > 30.0  # ambiguous without the probe
    below = reach_of(math.nextafter(weight_high, 0.0)).decide(perception, weighing, reading)
    assert (below.action, below.decided_by, below.zone) == (SET_ASIDE, "rule", "ambiguous")
    at = reach_of(weight_high).decide(perception, weighing, reading)
    assert (at.action, at.decided_by, at.zone) == (BIN, "rule", "surely_under")


def test_the_policy_file_gives_a_reach_of_45_ml():
    # Chosen 2026-11-17, from the sweep of thick drinks and frozen contents.
    assert CONFIG.probe_reach_ml == 45.0


def test_a_liquid_reading_under_the_threshold_bins_an_ambiguous_bottle():
    perception, weighing = exact_perception(BIG, 10.0), weighed(BIG, 10.0)
    assert UNLIMITED.decide(perception, weighing).zone == "ambiguous"
    verdict = UNLIMITED.decide(perception, weighing, ProbeReadings(liquid_ml=(5.0, 15.0)))
    assert (verdict.action, verdict.zone) == (BIN, "surely_under")
    assert verdict.bounds_ml == (5.0, 15.0)


def test_a_liquid_reading_that_reaches_the_threshold_does_not_bin():
    perception, weighing = exact_perception(BIG, 25.0), weighed(BIG, 25.0)
    verdict = UNLIMITED.decide(perception, weighing, ProbeReadings(liquid_ml=(20.0, 30.0)))
    assert (verdict.action, verdict.zone) == (SET_ASIDE, "ambiguous")
    assert verdict.bounds_ml[1] == 30.0


def test_a_liquid_reading_the_weight_contradicts_is_a_veto():
    perception, weighing = exact_perception(BIG, 0.0), weighed(BIG, 0.0)
    verdict = POLICY.decide(perception, weighing, ProbeReadings(liquid_ml=(500.0, 600.0)))
    assert (verdict.action, verdict.decided_by) == (SET_ASIDE, "veto:inconsistent_weight_and_probe")


def test_the_assessment_keeps_the_weights_interval_beside_the_fused_one():
    perception, weighing = exact_perception(BIG, 10.0), weighed(BIG, 10.0)
    by_hand = liquid_bounds_ml(weighing.mass_kg, empty_mass_bounds(perception.dims, PET), RULE)
    alone = UNLIMITED.assess(perception, weighing)
    assert alone.weight_ml == alone.v_bounds == by_hand
    fused = UNLIMITED.assess(perception, weighing, ProbeReadings(liquid_ml=(5.0, 15.0)))
    assert fused.weight_ml == by_hand
    assert fused.v_bounds == intersect_ml(by_hand, (5.0, 15.0)) == (5.0, 15.0)


SIZES = st.sampled_from(["pet_250ml", "pet_500ml", "pet_1000ml", "pet_2000ml"])


def interval(low, high):
    return st.tuples(st.floats(low, high), st.floats(low, high)).map(lambda p: tuple(sorted(p)))


@st.composite
def cases(draw):
    bottle = sample_bottle(
        FAMILY, draw(SIZES), PET, np.random.default_rng(draw(st.integers(0, 50)))
    )
    volume = draw(st.floats(0.0, 200.0))
    weighing = Weighing(
        mass_kg=bottle.empty_mass_kg + volume / 1000 + draw(st.floats(-0.04, 0.0)),
        stable=draw(st.booleans()),
    )
    perception = exact_perception(bottle, volume)
    if draw(st.booleans()):
        perception = dataclasses.replace(perception, material="glass")
    readings = ProbeReadings(
        side_wall_mm=draw(st.none() | interval(0.0, 1.0)),
        liquid_ml=draw(st.none() | interval(0.0, 500.0)),
    )
    return perception, weighing, readings


@settings(max_examples=300, derandomize=True, deadline=None)
@given(cases())
def test_no_probe_reading_moves_a_bottle_out_of_the_surely_over_zone(case):
    perception, weighing, readings = case
    before = POLICY.assess(perception, weighing)
    if before.zone is not Zone.SURELY_OVER:
        return
    after = POLICY.decide(perception, weighing, readings)
    assert after.action == SET_ASIDE


@settings(max_examples=300, derandomize=True, deadline=None)
@given(cases())
def test_no_probe_reading_removes_a_veto_the_policy_gave_without_it(case):
    perception, weighing, readings = case
    before = POLICY.decide(perception, weighing)
    if not before.decided_by.startswith("veto:"):
        return
    after = POLICY.decide(perception, weighing, readings)
    assert after.action == SET_ASIDE and after.decided_by.startswith("veto:")
