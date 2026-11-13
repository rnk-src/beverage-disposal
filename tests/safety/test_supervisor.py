"""The safety supervisor: each trip has a case that must trip and a near-miss that must not."""

import math

import numpy as np
import pytest
from support.observations import observation
from support.paths import ROOT

from beverage_disposal.core.interfaces import Observation
from beverage_disposal.core.kinematics import HOME
from beverage_disposal.core.registry import load_registry
from beverage_disposal.core.safety_config import load_safety_limits
from beverage_disposal.core.supervisor import (
    Outcome,
    RecordingIndicators,
    Supervisor,
)

LIMITS = load_safety_limits(ROOT / "configs" / "safety.toml")
REGISTRY = load_registry(ROOT / "configs" / "containers.toml")
CAN = REGISTRY.lookup("can_12floz")


def fresh():
    indicators = RecordingIndicators()
    return Supervisor(LIMITS, indicators), indicators


def good_observation(**changes):
    base = observation(HOME, 0.5, (0.27, 0.0, 0.061), (0.17, 0.15, 0.02))
    return Observation(**{**base.__dict__, **changes})


def test_the_limits_file_has_the_limits_the_comments_explain():
    assert (LIMITS.max_container_diameter, LIMITS.max_container_mass) == (0.075, 0.60)
    assert LIMITS.max_arm_force == 20.0 and LIMITS.max_episode_seconds == 20.0


@pytest.mark.parametrize(
    ("diameter", "trips"), [(0.0749, False), (0.0751, True)], ids=["near-miss", "trip"]
)
def test_a_container_wider_than_the_pads_can_take_is_refused(diameter, trips):
    supervisor, _ = fresh()
    decision = supervisor.check_container(CAN, diameter=diameter, mass_kg=0.05)
    assert (decision.outcome is Outcome.SAFE_STOP) == trips
    assert (decision.event is not None and decision.event.code == "too_wide") == trips


@pytest.mark.parametrize("mass", [0.599, 0.601], ids=["near-miss", "trip"])
def test_a_container_heavier_than_the_arm_should_lift_is_refused(mass):
    supervisor, _ = fresh()
    decision = supervisor.check_container(CAN, diameter=0.066, mass_kg=mass)
    assert (decision.outcome is Outcome.SAFE_STOP) == (mass > 0.6)


def test_a_container_of_an_unknown_type_is_set_aside_not_stopped():
    supervisor, _ = fresh()
    decision = supervisor.check_container(None, diameter=0.066, mass_kg=0.05)
    assert decision.outcome is Outcome.SET_ASIDE and decision.event.code == "unknown_type"


def test_a_good_container_proceeds_without_a_word():
    supervisor, indicators = fresh()
    decision = supervisor.check_container(CAN, diameter=0.066, mass_kg=0.05)
    assert decision.outcome is Outcome.PROCEED and decision.event is None
    assert indicators.log == []


@pytest.mark.parametrize("gain", [0.0271, 0.0269], ids=["near-miss", "trip"])
def test_a_test_lift_that_does_not_raise_the_container_enough_stops_the_pick(gain):
    supervisor, _ = fresh()
    decision = supervisor.check_test_lift(gain_m=gain, asked_m=0.03)
    assert (decision.outcome is Outcome.SAFE_STOP) == (gain < 0.9 * 0.03)


@pytest.mark.parametrize("force", [19.9, 20.1], ids=["near-miss", "trip"])
def test_an_arm_pressing_too_hard_is_stopped(force):
    supervisor, _ = fresh()
    decision = supervisor.check_forces(arm_force_n=force, grip_torque_nm=0.6)
    assert (decision.outcome is Outcome.SAFE_STOP) == (force > 20.0)


@pytest.mark.parametrize("torque", [0.99, 1.01], ids=["near-miss", "trip"])
def test_a_squeeze_beyond_the_gripper_limit_is_stopped(torque):
    supervisor, _ = fresh()
    decision = supervisor.check_forces(arm_force_n=0.0, grip_torque_nm=torque)
    assert (decision.outcome is Outcome.SAFE_STOP) == (torque > 1.0)


def test_an_observation_with_a_nan_or_an_infinity_is_not_trusted():
    supervisor, _ = fresh()
    assert supervisor.check_observation(good_observation()).outcome is Outcome.PROCEED
    for field in ["joints", "tip", "container", "goal"]:
        bad = np.array(getattr(good_observation(), field), dtype=float)
        bad[0] = math.nan
        assert (
            supervisor.check_observation(good_observation(**{field: bad})).outcome
            is Outcome.SAFE_STOP
        )
    inf = np.array(good_observation().container)
    inf[1] = math.inf
    assert (
        supervisor.check_observation(good_observation(container=inf)).outcome is Outcome.SAFE_STOP
    )
    assert (
        supervisor.check_observation(good_observation(gripper=math.nan)).outcome
        is Outcome.SAFE_STOP
    )


def test_a_missing_observation_is_a_sensor_dropout_and_stops():
    supervisor, _ = fresh()
    decision = supervisor.check_observation(None)
    assert decision.outcome is Outcome.SAFE_STOP and decision.event.code == "sensor_dropout"


@pytest.mark.parametrize("seconds", [19.9, 20.1], ids=["near-miss", "trip"])
def test_an_episode_that_runs_too_long_is_stopped(seconds):
    supervisor, _ = fresh()
    assert (supervisor.check_time(seconds).outcome is Outcome.SAFE_STOP) == (seconds > 20.0)


def test_a_container_that_is_not_where_it_was_seen_stops_the_pick():
    supervisor, _ = fresh()
    seen = np.array([0.27, 0.0, 0.061])
    assert (
        supervisor.check_container_position(seen, seen + [0.005, 0.0, 0.0]).outcome
        is Outcome.PROCEED
    )
    gone = supervisor.check_container_position(seen, seen + [0.05, 0.0, 0.0])
    assert gone.outcome is Outcome.SAFE_STOP and gone.event.code == "container_moved"
    assert supervisor.check_container_position(seen, None).event.code == "container_missing"


def test_a_check_that_raises_ends_in_a_safe_stop_and_never_escapes():
    supervisor, indicators = fresh()

    def broken():
        raise RuntimeError("bug in a check")

    decision = supervisor.guard(broken)
    assert decision.outcome is Outcome.SAFE_STOP and decision.event.code == "supervisor_fault"
    assert "bug in a check" in decision.event.message
    assert indicators.log  # the fault was reported


def test_a_guard_passes_a_normal_decision_through_untouched():
    supervisor, _ = fresh()
    decision = supervisor.guard(lambda: supervisor.check_time(1.0))
    assert decision.outcome is Outcome.PROCEED


def test_a_non_finite_action_is_replaced_by_standing_still_and_stops():
    supervisor, _ = fresh()
    action, decision = supervisor.filter_action(np.array([0.5, math.nan, 0.0, 1.0]))
    assert np.all(action == 0.0) and decision.outcome is Outcome.SAFE_STOP


def test_an_action_out_of_range_is_clipped_and_not_a_fault():
    supervisor, _ = fresh()
    action, decision = supervisor.filter_action(np.array([3.0, -2.0, 0.5, 0.0]))
    assert action == pytest.approx([1.0, -1.0, 0.5, 0.0])
    assert decision.outcome is Outcome.PROCEED


@pytest.mark.parametrize(
    "trip",
    [
        lambda s: s.check_container(CAN, diameter=0.08, mass_kg=0.05),
        lambda s: s.check_container(None, diameter=0.066, mass_kg=0.05),
        lambda s: s.check_test_lift(gain_m=0.0, asked_m=0.03),
        lambda s: s.check_forces(arm_force_n=50.0, grip_torque_nm=0.6),
        lambda s: s.check_observation(None),
        lambda s: s.check_time(30.0),
    ],
    ids=["wide", "unknown", "lift", "force", "dropout", "time"],
)
def test_every_trip_is_reported_by_a_light_a_sound_and_a_message(trip):
    supervisor, indicators = fresh()
    decision = trip(supervisor)
    assert decision.outcome is not Outcome.PROCEED
    kinds = [kind for kind, _ in indicators.log]
    assert sorted(kinds) == ["light", "message", "sound"]
    assert dict(indicators.log)["message"] == decision.event.message


def test_a_stop_lights_red_and_a_set_aside_lights_amber():
    supervisor, indicators = fresh()
    supervisor.check_container(CAN, diameter=0.08, mass_kg=0.05)
    supervisor.check_container(None, diameter=0.066, mass_kg=0.05)
    lights = [value for kind, value in indicators.log if kind == "light"]
    assert lights == ["red", "amber"]


def test_the_supervisor_keeps_the_events_in_the_order_they_happened():
    supervisor, _ = fresh()
    supervisor.check_time(30.0)
    supervisor.check_observation(None)
    assert [e.code for e in supervisor.events] == ["timeout", "sensor_dropout"]
