"""Servo step responses in MuJoCo, and the reference file Gazebo will be matched against (D-16)."""

import importlib.util
import json

import numpy as np
import pytest
from support.paths import ROOT
from support.so101 import JOINTS, MJCF, SO101

from beverage_disposal.core.kinematics import HOME
from beverage_disposal.sim.actuator_response import measure_step_responses
from beverage_disposal.sim.arm import load_arm

REFERENCE = ROOT / "configs" / "actuator_step_response.json"
GENERATOR = ROOT / "scripts" / "measure_actuators.py"


@pytest.fixture(scope="module")
def arm():
    return load_arm(SO101 / MJCF)


@pytest.fixture(scope="module")
def measurement(arm):
    return measure_step_responses(arm)


def test_every_joint_has_a_response_and_the_protocol_that_produced_it(measurement):
    assert list(measurement["joints"]) == JOINTS
    protocol = measurement["protocol"]
    assert protocol["timestep"] == 0.002
    assert protocol["control_period"] == 0.05
    assert protocol["start_pose"] == list(HOME)
    assert protocol["step_size"] == 0.2


def test_the_servo_parameters_in_the_record_are_the_mjcfs(measurement):
    # kp 998.22, kv 2.731, force limit +-3.35 N m, for all six.
    for gains in measurement["protocol"]["servo"].values():
        assert gains == {"kp": 998.22, "kv": 2.731, "force_limit": 3.35}


def test_every_joint_follows_a_step_fast_with_moderate_overshoot_and_no_sag(measurement):
    # Bounds fixed after the first measurements (1.6 log): rise 50 to 56 ms, overshoot 18 to 33 %,
    # settling 150 to 222 ms, steady-state error under 0.5 mrad. Generous enough to survive a
    # timestep tweak, tight enough to catch a servo that has become sluggish or unstable.
    for name, result in measurement["joints"].items():
        metrics = result["metrics"]
        assert 0.03 < metrics["rise_time"] < 0.10, name
        assert metrics["overshoot_percent"] < 40, name
        assert metrics["settling_time"] < 0.30, name
        assert metrics["steady_state_error"] < 1e-3, name


def test_the_committed_reference_file_is_what_the_measurement_gives_today(measurement):
    committed = json.loads(REFERENCE.read_text())
    assert committed["protocol"] == measurement["protocol"]
    for name in JOINTS:
        for metric, value in measurement["joints"][name]["metrics"].items():
            expected = committed["joints"][name]["metrics"][metric]
            assert value == pytest.approx(expected, rel=1e-6, abs=1e-9), (name, metric)


def test_the_script_writes_the_reference_file_byte_for_byte(tmp_path):
    spec = importlib.util.spec_from_file_location("measure_actuators", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "reference.json"
    module.main(["--out", str(out)])
    assert json.loads(out.read_text()) == json.loads(REFERENCE.read_text())


def test_the_alternative_servo_numbers_from_joints_properties_sag_and_never_settle(arm):
    # joints_properties.xml says kp 17.8, kv 0. As N m / rad in MuJoCo they leave the lift joint
    # 27 mrad short and ringing for good, which is why the MJCF's gains are kept (1.6 log).
    worse = load_arm(SO101 / MJCF)
    for actuator in range(worse.nu):
        worse.actuator_gainprm[actuator, 0] = 17.8
        worse.actuator_biasprm[actuator, 1:3] = [-17.8, 0.0]
    lift = measure_step_responses(worse)["joints"]["shoulder_lift"]["metrics"]
    assert lift["steady_state_error"] > 10e-3
    assert np.isinf(lift["settling_time"])
