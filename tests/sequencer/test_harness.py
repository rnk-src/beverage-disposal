"""The evaluation harness: a grid of seeded runs and the results record around it."""

import json

import pytest
from support.so101 import MJCF, SO101
from test_sequencer import CAN, FULLNESS, LIMITS, REGISTRY, SCENE, TILT

from beverage_disposal.core.sequencer import IdealSkill
from beverage_disposal.sim.harness import evaluate
from beverage_disposal.sim.skills import SimSkill

SETUP = dict(scene=SCENE, registry=REGISTRY, fullness=FULLNESS, tilt=TILT, limits=LIMITS)


def grid(skill, volumes=(0.0, 31.0), n=3, **options):
    return evaluate(skill, volumes_ml=volumes, trials=n, container=CAN, **SETUP, **options)


def test_the_results_record_names_the_seeds_and_counts_each_volume():
    results = grid(IdealSkill())
    assert results["seeds"] == {"0": [0, 1, 2], "31": [0, 1, 2]}
    assert results["summary"]["runs"] == 6
    assert results["summary"]["binned_at_or_above_threshold"]["31"]["n"] == 3
    json.dumps(results)


def test_the_record_holds_what_is_needed_to_reproduce_the_run():
    results = grid(IdealSkill())
    assert set(results["provenance"]) >= {"git_sha", "config_hash", "versions"}
    assert results["provenance"]["config_hash"] == grid(IdealSkill())["provenance"]["config_hash"]


def test_the_same_seeds_give_the_same_summary():
    assert grid(IdealSkill())["summary"] == grid(IdealSkill())["summary"]


def test_a_hundred_percent_misbinning_rule_shows_up_in_the_summary():
    import dataclasses

    broken = dataclasses.replace(
        FULLNESS, rule=dataclasses.replace(FULLNESS.rule, threshold_ml=1000.0)
    )
    options = {**SETUP, "fullness": broken}
    results = evaluate(IdealSkill(), volumes_ml=(35.0,), trials=3, container=CAN, **options)
    assert results["summary"]["binned_at_or_above_threshold"]["35"]["k"] == 3


@pytest.mark.medium
def test_the_simulated_skill_runs_through_the_harness():
    results = grid(SimSkill(SCENE, SO101 / MJCF), volumes=(0.0,), n=2)
    assert results["summary"]["pick_and_place"]["n"] == 2
