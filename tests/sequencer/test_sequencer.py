"""The sequencer: pick to the pad, weigh, decide, pick again to the bin or the set-aside area."""

import dataclasses

import numpy as np
from support.paths import ROOT

from beverage_disposal.core.coinft import CoinFT
from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.layout import goals
from beverage_disposal.core.pad import PadSensor
from beverage_disposal.core.registry import load_registry
from beverage_disposal.core.safety_config import load_safety_limits
from beverage_disposal.core.sequencer import (
    BINNED,
    SET_ASIDE_AREA,
    STOPPED,
    IdealSkill,
    Item,
    Sequencer,
    SkillResult,
)
from beverage_disposal.core.supervisor import RecordingIndicators, Supervisor
from beverage_disposal.core.tilt_config import load_tilt_config

CONFIGS = ROOT / "configs"
SCENE = load_scene_config(CONFIGS / "scene.toml")
REGISTRY = load_registry(CONFIGS / "containers.toml")
FULLNESS = load_fullness_config(CONFIGS / "fullness.toml")
TILT = load_tilt_config(CONFIGS / "tilt.toml")
LIMITS = load_safety_limits(CONFIGS / "safety.toml")
CAN = REGISTRY.lookup("can_12floz")
BOTTLE = REGISTRY.lookup("bottle_500ml")
GOALS = goals(SCENE)


class RecordingSkill(IdealSkill):
    """An ideal skill that remembers its calls, and can be told to fail on a given call."""

    def __init__(self, fail_on=None):
        self.calls = []
        self._fail_on = fail_on

    def pick_and_place(self, source, goal, *, item, seed):
        self.calls.append((source, goal.name))
        if self._fail_on == len(self.calls):
            return SkillResult(False, "the grasp failed")
        return SkillResult(True, "ok")


def item(record=CAN, volume_ml=0.0, density=1.0, known=True, **changes):
    base = Item(
        type_id=record.type_id if known else None,
        spec=record.spec,
        volume_ml=volume_ml,
        density=density,
        empty_mass_kg=record.spec.empty_mass,
        cap_confirmed=None if not record.cap_check_required else True,
    )
    return dataclasses.replace(base, **changes)


def sequencer(skill=None, *, use_tilt=True, registry=REGISTRY, seed=0):
    skill = skill or RecordingSkill()
    indicators = RecordingIndicators()
    rng = np.random.default_rng(seed)
    machine = Sequencer(
        scene=SCENE,
        registry=registry,
        fullness=FULLNESS,
        tilt=TILT,
        supervisor=Supervisor(LIMITS, indicators),
        skill=skill,
        pad=PadSensor(FULLNESS.pad, rng),
        coinft=CoinFT(TILT.sensor, rng),
        use_tilt=use_tilt,
    )
    return machine, skill, indicators


def test_an_empty_known_can_goes_to_the_pad_is_weighed_and_goes_to_the_bin():
    machine, skill, _ = sequencer()
    run = machine.process(item(volume_ml=0.0), seed=0)
    assert run.destination == BINNED
    assert [goal for _, goal in skill.calls] == ["pad", "bin"]
    assert run.verdict.action == "bin"


def test_a_known_can_of_thirty_one_millilitres_goes_to_the_set_aside_area():
    machine, skill, _ = sequencer()
    run = machine.process(item(volume_ml=31.0), seed=0)
    assert run.destination == SET_ASIDE_AREA
    assert [goal for _, goal in skill.calls] == ["pad", "set_aside"]


def test_the_first_pick_is_from_the_spawn_box_and_the_second_from_the_pad():
    machine, skill, _ = sequencer()
    machine.process(item(), seed=0)
    first, second = (source for source, _ in skill.calls)
    assert first is None  # picked where it was spawned, chosen by the skill from the seed
    pad = GOALS["pad"]
    assert (second.x, second.y) == (pad.center_x, pad.center_y)
    assert second.surface_z == pad.rest_z


def test_a_bottle_whose_cap_is_not_confirmed_is_set_aside_and_one_with_its_cap_is_binned():
    machine, _, _ = sequencer()
    assert machine.process(item(BOTTLE, 0.0), seed=0).destination == BINNED
    unconfirmed = machine.process(item(BOTTLE, 0.0, cap_confirmed=None), seed=0)
    assert unconfirmed.destination == SET_ASIDE_AREA


def test_a_container_too_wide_for_the_pads_stops_the_run_before_any_pick():
    machine, skill, indicators = sequencer()
    wide = dataclasses.replace(CAN.spec, diameter=0.08)
    run = machine.process(item(spec=wide), seed=0)
    assert run.destination == STOPPED and skill.calls == []
    assert run.events[0].code == "too_wide"
    assert ("light", "red") in indicators.log


def test_a_failed_first_pick_stops_the_run_and_reports_it():
    machine, skill, _ = sequencer(RecordingSkill(fail_on=1))
    run = machine.process(item(), seed=0)
    assert run.destination == STOPPED and len(skill.calls) == 1
    assert run.events[-1].code == "pick_failed"


def test_a_failed_second_pick_stops_the_run_with_the_container_left_on_the_pad():
    machine, skill, _ = sequencer(RecordingSkill(fail_on=2))
    run = machine.process(item(), seed=0)
    assert run.destination == STOPPED and len(skill.calls) == 2
    assert run.events[-1].code == "place_failed"


def test_a_container_of_an_unknown_type_is_met_with_an_enrolment_request_and_the_tilt_test():
    machine, _, indicators = sequencer()
    run = machine.process(item(known=False, volume_ml=0.0), seed=0)
    assert run.destination == BINNED  # the tilt test bounds an empty can under 30 mL
    assert [r.type_id for r in machine.registry.pending] == ["unrecognised"]
    assert any(event.code == "unknown_type" for event in run.events)
    assert ("light", "amber") in indicators.log


def test_an_unknown_container_of_twenty_five_millilitres_is_set_aside_by_the_strict_tilt_test():
    machine, _, _ = sequencer()
    assert machine.process(item(known=False, volume_ml=25.0), seed=0).destination == SET_ASIDE_AREA


def test_an_unknown_container_heavier_than_the_limit_is_set_aside_without_a_tilt():
    machine, _, _ = sequencer()
    heavy = item(known=False, volume_ml=0.0, empty_mass_kg=0.10)
    run = machine.process(heavy, seed=0)
    assert run.destination == SET_ASIDE_AREA and "heavy" in run.verdict.reason


def test_without_the_coinft_an_unknown_container_is_binned_only_if_all_of_it_is_under_the_limit():
    machine, _, _ = sequencer(use_tilt=False)
    # A 14 g empty can weighs under the 26.5 g limit; a 30 g empty one does not, whatever is inside.
    assert machine.process(item(known=False, volume_ml=0.0), seed=0).destination == BINNED
    heavier = item(known=False, volume_ml=0.0, empty_mass_kg=0.030)
    assert machine.process(heavier, seed=0).destination == SET_ASIDE_AREA


def test_the_same_seed_gives_the_same_run():
    first = sequencer(seed=3)[0].process(item(volume_ml=20.0), seed=3)
    second = sequencer(seed=3)[0].process(item(volume_ml=20.0), seed=3)
    assert (first.destination, first.verdict.net_g) == (second.destination, second.verdict.net_g)


def test_a_run_says_whether_the_rule_binned_something_it_should_not_have():
    machine, _, _ = sequencer()
    assert not machine.process(item(volume_ml=0.0), seed=0).misbinned
    # Force a wrong decision: a rule that thinks 100 mL is the limit lets 40 mL through.
    loose = dataclasses.replace(
        FULLNESS, rule=dataclasses.replace(FULLNESS.rule, threshold_ml=100.0)
    )
    sloppy = dataclasses.replace(machine, fullness=loose)
    run = sloppy.process(item(volume_ml=40.0), seed=0)
    assert run.destination == BINNED
    assert run.misbinned  # judged against the 30 mL requirement, not the loose rule's own number
