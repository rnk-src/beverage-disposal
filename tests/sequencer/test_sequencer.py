"""The sequencer: a container is put on the pad, weighed and judged there, then carried once."""

import dataclasses

import numpy as np
from support.paths import ROOT

from beverage_disposal.core.bottle_family import container_spec_for, load_family, sample_bottle
from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.layout import goals
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.pad import PadSensor
from beverage_disposal.core.perception import load_perception_noise
from beverage_disposal.core.policy import EmptinessPolicy, load_policy_config
from beverage_disposal.core.registry import load_registry
from beverage_disposal.core.safety_config import load_safety_limits
from beverage_disposal.core.sequencer import (
    BINNED,
    INTAKE_SCATTER,
    SET_ASIDE_AREA,
    STOPPED,
    IdealSkill,
    Item,
    Sequencer,
    SkillResult,
)
from beverage_disposal.core.supervisor import RecordingIndicators, Supervisor

CONFIGS = ROOT / "configs"
SCENE = load_scene_config(CONFIGS / "scene.toml")
REGISTRY = load_registry(CONFIGS / "containers.toml")
FULLNESS = load_fullness_config(CONFIGS / "fullness.toml")
LIMITS = load_safety_limits(CONFIGS / "safety.toml")
CAN = REGISTRY.lookup("can_12floz")
BOTTLE = REGISTRY.lookup("bottle_500ml")
FAMILY = load_family(CONFIGS / "bottles.toml")
POLICY_CONFIG = load_policy_config(CONFIGS / "policy.toml")
NOISE = load_perception_noise(CONFIGS / "perception.toml")
PET = load_materials(CONFIGS / "materials.toml")["PET"]
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


def sequencer(skill=None, *, registry=REGISTRY, seed=0, policy=None, noise=None):
    skill = skill or RecordingSkill()
    indicators = RecordingIndicators()
    rng = np.random.default_rng(seed)
    machine = Sequencer(
        scene=SCENE,
        registry=registry,
        fullness=FULLNESS,
        supervisor=Supervisor(LIMITS, indicators),
        skill=skill,
        pad=PadSensor(FULLNESS.pad, rng),
        policy=policy,
        perception_noise=noise,
    )
    return machine, skill, indicators


def pet_item(volume_ml=0.0, size="pet_500ml", seed=0):
    """A drawn PET bottle as an item: perception gives no registry type for it."""
    bottle = sample_bottle(FAMILY, size, PET, np.random.default_rng(seed))
    spec = container_spec_for(
        bottle, FAMILY.classes[size], friction=CAN.spec.friction, torsional_friction=0.01
    )
    return Item(
        type_id=None,
        spec=spec,
        volume_ml=volume_ml,
        density=1.0,
        empty_mass_kg=bottle.empty_mass_kg,
        cap_confirmed=None,
        bottle=bottle,
        material="PET",
    )


def test_an_empty_known_can_is_weighed_on_the_pad_and_carried_once_to_the_bin():
    machine, skill, _ = sequencer()
    run = machine.process(item(volume_ml=0.0), seed=0)
    assert run.destination == BINNED
    assert [goal for _, goal in skill.calls] == ["bin"]
    assert run.verdict.action == "bin"


def test_a_known_can_of_thirty_one_millilitres_is_carried_once_to_the_set_aside_area():
    machine, skill, _ = sequencer()
    run = machine.process(item(volume_ml=31.0), seed=0)
    assert run.destination == SET_ASIDE_AREA
    assert [goal for _, goal in skill.calls] == ["set_aside"]


def test_the_one_pick_is_from_the_pad_where_a_person_put_the_container():
    pad = SCENE.pad
    spots = []
    for seed in range(20):
        machine, skill, _ = sequencer()
        machine.process(item(), seed=seed)
        ((source, _),) = skill.calls
        assert source.surface_z == pad.height
        spots.append((source.x - pad.center_x, source.y - pad.center_y))
    spots = np.array(spots)
    # A person does not hit the centre: each seed is off by up to the scatter, never more.
    assert np.abs(spots).max() <= INTAKE_SCATTER
    assert np.abs(spots).max() > INTAKE_SCATTER / 2 and len({tuple(s) for s in spots}) == 20


def test_a_bottle_whose_cap_is_not_confirmed_is_set_aside_and_one_with_its_cap_is_binned():
    machine, _, _ = sequencer()
    assert machine.process(item(BOTTLE, 0.0), seed=0).destination == BINNED
    unconfirmed = machine.process(item(BOTTLE, 0.0, cap_confirmed=None), seed=0)
    assert unconfirmed.destination == SET_ASIDE_AREA


def test_a_grasp_too_wide_for_the_pads_stops_the_run_before_any_pick():
    machine, skill, indicators = sequencer()
    wide = dataclasses.replace(CAN.spec, grasp_diameter=0.08)
    run = machine.process(item(spec=wide), seed=0)
    assert run.destination == STOPPED and skill.calls == []
    assert run.events[0].code == "too_wide"
    assert ("light", "red") in indicators.log


def test_a_wide_body_taken_by_a_narrow_neck_is_not_too_wide():
    machine, skill, _ = sequencer()
    neck = dataclasses.replace(CAN.spec, diameter=0.105, grasp_diameter=0.027)
    assert machine.process(item(spec=neck), seed=0).destination != STOPPED


def test_a_container_over_the_payload_is_never_handed_to_the_skill():
    """Catches a lift of a container the arm cannot carry: it is decided from the pad's weight."""
    machine, skill, _ = sequencer()
    run = machine.process(item(volume_ml=600.0), seed=0)  # 0.61 kg with the can
    assert run.destination == STOPPED and skill.calls == []
    assert run.events[-1].code == "too_heavy_remove_by_hand"
    assert run.verdict is None


def test_a_container_that_overloads_the_pad_is_never_handed_to_the_skill():
    machine, skill, _ = sequencer()
    run = machine.process(item(volume_ml=2100.0), seed=0)
    assert run.destination == STOPPED and skill.calls == []
    assert run.events[-1].code == "too_heavy_remove_by_hand"


def test_water_on_the_outside_counts_in_the_weighed_mass():
    # 29.5 g of water on the outside of an empty can reads like a can that is not empty.
    machine, _, _ = sequencer()
    run = machine.process(item(volume_ml=0.0, wet_g=29.5), seed=0)
    assert run.destination == SET_ASIDE_AREA
    assert run.verdict.net_g > 26.5


def test_a_failed_carry_stops_the_run_with_the_container_left_where_it_fell():
    machine, skill, _ = sequencer(RecordingSkill(fail_on=1))
    run = machine.process(item(), seed=0)
    assert run.destination == STOPPED and len(skill.calls) == 1
    assert run.events[-1].code == "place_failed"
    assert run.verdict.action == "bin"  # the decision was made before the carry


def test_an_empty_container_of_an_unknown_type_is_set_aside_not_binned():
    """Catches an unknown type being binned on a weight guess: its material is not known."""
    machine, skill, indicators = sequencer()
    run = machine.process(item(known=False, volume_ml=0.0), seed=0)
    assert run.destination == SET_ASIDE_AREA
    assert [goal for _, goal in skill.calls] == ["set_aside"]
    assert run.verdict.reason == "unknown container type"
    assert [r.type_id for r in machine.registry.pending] == ["unrecognised"]
    assert run.events[0].code == "unknown_type"
    assert ("light", "amber") in indicators.log


def test_a_pet_bottle_is_set_aside_while_there_is_no_emptiness_policy():
    machine, skill, _ = sequencer()
    run = machine.process(pet_item(volume_ml=0.0), seed=0)
    assert run.destination == SET_ASIDE_AREA
    assert [goal for _, goal in skill.calls] == ["set_aside"]


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


def policy(sizes=tuple(FAMILY.classes)):
    return EmptinessPolicy(FULLNESS.rule, PET, FAMILY, POLICY_CONFIG, sizes=sizes)


def test_with_a_policy_an_empty_pet_bottle_is_binned_and_a_half_full_one_set_aside():
    machine, skill, _ = sequencer(policy=policy())
    empty = machine.process(pet_item(volume_ml=0.0), seed=0)
    assert empty.destination == BINNED and empty.verdict.zone == "surely_under"
    half = machine.process(pet_item(volume_ml=250.0), seed=0)
    assert half.destination == SET_ASIDE_AREA and half.verdict.zone == "surely_over"
    assert [goal for _, goal in skill.calls] == ["bin", "set_aside"]


def test_a_pet_bottle_judged_by_the_policy_is_not_sent_for_enrolment():
    machine, _, indicators = sequencer(policy=policy())
    run = machine.process(pet_item(volume_ml=0.0), seed=0)
    assert machine.registry.pending == REGISTRY.pending
    assert run.events == [] and indicators.log == []


def test_a_policy_veto_is_reported_under_its_code_and_the_bottle_set_aside():
    machine, _, indicators = sequencer(policy=policy(sizes=()))
    run = machine.process(pet_item(volume_ml=0.0), seed=0)
    assert run.destination == SET_ASIDE_AREA
    assert run.verdict.decided_by == "veto:unsupported_size"
    assert [event.code for event in run.events] == ["unsupported_size"]
    assert ("light", "amber") in indicators.log


def test_the_policy_does_not_touch_a_can_or_an_unknown_container_that_is_not_pet():
    machine, _, _ = sequencer(policy=policy())
    assert machine.process(item(volume_ml=0.0), seed=0).verdict.decided_by == "rule"
    unknown = machine.process(item(known=False, volume_ml=0.0), seed=0)
    assert unknown.verdict.reason == "unknown container type"


def test_with_camera_noise_the_bottle_is_judged_on_what_the_camera_reports():
    def most_liquid(noise):
        machine, _, _ = sequencer(policy=policy(), noise=noise)
        return machine.process(pet_item(volume_ml=0.0), seed=0).verdict.bounds_ml[1]

    # The wider the camera's intervals, the lighter the bottle may be and the more it may hold.
    vague = dataclasses.replace(NOISE, interval_k=10.0)
    assert most_liquid(None) < most_liquid(NOISE) < most_liquid(vague)


def test_the_supervisor_gets_the_grasp_width_the_camera_reports_as_an_interval():
    # A 26 mm neck seen with +-81 mm may be far too wide for the pads: stop before any lift.
    blind = dataclasses.replace(NOISE, interval_k=40.0)
    machine, skill, _ = sequencer(policy=policy(), noise=blind)
    run = machine.process(pet_item(volume_ml=0.0), seed=0)
    assert run.destination == STOPPED and skill.calls == []
    assert run.events[0].code == "too_wide"


def test_the_cameras_look_follows_the_seed():
    def bounds(seed):
        machine, _, _ = sequencer(policy=policy(), noise=NOISE)
        return machine.process(pet_item(volume_ml=0.0), seed=seed).verdict.bounds_ml

    assert bounds(1) == bounds(1) and bounds(1) != bounds(2)
