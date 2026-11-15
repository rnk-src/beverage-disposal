"""The evaluation harness: a grid of seeded runs and the results record around it."""

import json

import pytest
from support.so101 import MJCF, SO101
from test_sequencer import CAN, FULLNESS, LIMITS, REGISTRY, SCENE

from beverage_disposal.core.sequencer import IdealSkill, SkillResult
from beverage_disposal.sim.harness import evaluate
from beverage_disposal.sim.skills import SimSkill

SETUP = dict(scene=SCENE, registry=REGISTRY, fullness=FULLNESS, limits=LIMITS)


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


def test_a_grid_of_unknown_containers_bins_none_of_them():
    """Catches the unknown-type path binning anything, at any volume."""
    results = grid(IdealSkill(), volumes=(0.0, 31.0), known=False)
    assert results["summary"]["runs"] == 6
    assert all(row["destination"] == "set_aside" for row in results["runs"])


class FailingSkill:
    def pick_and_place(self, source, goal, *, item, seed):
        return SkillResult(False, "time ran out")


def test_the_record_has_one_row_per_run_saying_how_it_ended():
    rows = grid(IdealSkill())["runs"]
    assert [(row["volume_ml"], row["seed"]) for row in rows] == [
        (0.0, 0),
        (0.0, 1),
        (0.0, 2),
        (31.0, 0),
        (31.0, 1),
        (31.0, 2),
    ]
    assert {row["destination"] for row in rows if row["volume_ml"] == 0.0} == {"binned"}
    assert all(row["stop"] is None for row in rows)


def test_a_stopped_run_names_its_stop_code_and_the_skills_reason():
    results = grid(FailingSkill(), volumes=(0.0,), n=2)
    for row in results["runs"]:
        assert row["destination"] == "stopped"
        assert row["stop"] == {
            "code": "pick_failed",
            "message": "the first pick failed: time ran out",
        }
    assert results["summary"]["stops"] == {"pick_failed": 2}
    json.dumps(results)


@pytest.mark.medium
def test_the_simulated_skill_runs_through_the_harness():
    results = grid(SimSkill(SCENE, SO101 / MJCF), volumes=(0.0,), n=2)
    assert results["summary"]["pick_and_place"]["n"] == 2
