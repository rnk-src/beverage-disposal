"""The rule-only emptiness policy: each veto alone, their order, and the three zones."""

import dataclasses
import tomllib

import numpy as np
import pytest
from support.paths import ROOT

from beverage_disposal.core.bottle_family import GateRecord, load_family, sample_bottle
from beverage_disposal.core.bounds import empty_mass_bounds
from beverage_disposal.core.fullness import BIN, SET_ASIDE
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.pad import Weighing
from beverage_disposal.core.perception import exact_perception
from beverage_disposal.core.policy import EmptinessPolicy, load_policy_config
from beverage_disposal.core.zones import liquid_bounds_ml

CONFIGS = ROOT / "configs"
RULE = load_fullness_config(CONFIGS / "fullness.toml").rule
FAMILY = load_family(CONFIGS / "bottles.toml")
PET = load_materials(CONFIGS / "materials.toml")["PET"]
CONFIG = load_policy_config(CONFIGS / "policy.toml")
ALL_SIZES = tuple(FAMILY.classes)
POLICY = EmptinessPolicy(RULE, PET, FAMILY, CONFIG, sizes=ALL_SIZES)
BOTTLE = sample_bottle(FAMILY, "pet_500ml", PET, np.random.default_rng(0))
BIG = sample_bottle(FAMILY, "pet_2000ml", PET, np.random.default_rng(0))


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
    assert CONFIG.allow_overturn is True and CONFIG.min_confidence == 0.8


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
    assert decide(policy=dataclasses.replace(POLICY, sizes=("pet_500ml",))).action == BIN


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
    passed = dataclasses.replace(
        FAMILY.classes["pet_500ml"], gate=GateRecord("passed", "2026-11-16", "")
    )
    family = dataclasses.replace(FAMILY, classes={**FAMILY.classes, "pet_500ml": passed})
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
