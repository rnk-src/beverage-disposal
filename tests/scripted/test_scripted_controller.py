"""The scripted pick-and-place as a state machine, driven by a kinematic stand-in for the arm.

The stand-in moves the tip by exactly the commanded step and holds the can when the gripper is
closed on it. It has no physics, so these tests check the order and the targets of the phases;
test_scripted_rollout.py checks the same script against the real simulation.
"""

import math

import numpy as np
import pytest
from support.observations import observation
from support.paths import ROOT

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.controller import MAX_DELTA
from beverage_disposal.core.ik import solve_ik
from beverage_disposal.core.kinematics import HOME, fk, load_chain
from beverage_disposal.core.layout import goals
from beverage_disposal.core.scripted import Phase, ScriptedController

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
GRIPPER_LOW, GRIPPER_HIGH = load_chain()["gripper_range"]
CAN = np.array([0.29, 0.0, 0.061])  # the can's centre at rest on the table
GOAL = goals(CONFIG)["set_aside"]
# Held against the fixed pad, the can's centre is 25.1 mm from the TCP along the closing direction.
HELD_OFFSET = 0.0251


def ready_joints():
    start = np.array(HOME, dtype=float)
    start[4] = CONFIG.grasp.wrist_roll
    return solve_ik(np.array([0.26, 0.0, 0.15]), math.radians(CONFIG.grasp.pitch_deg), start).angles


JOINTS = ready_joints()
CLOSING = fk(JOINTS)[:2, 0] / np.linalg.norm(fk(JOINTS)[:2, 0])


def angle_of(action):
    return GRIPPER_LOW + (action + 1) / 2 * (GRIPPER_HIGH - GRIPPER_LOW)


class KinematicWorld:
    """Tip, gripper and can, moved exactly as commanded."""

    def __init__(self, gripper_follows=True):
        self.tip = fk(JOINTS)[:3, 3]
        self.gripper = CONFIG.grasp.open_angle
        self.can = CAN.copy()
        self.held = False
        self.previous = np.zeros(4)

    def observe(self):
        return observation(
            JOINTS,
            self.gripper,
            self.can,
            (GOAL.center_x, GOAL.center_y, GOAL.release_z),
            previous_action=self.previous,
            tip=self.tip,
        )

    def apply(self, action):
        action = np.clip(action, -1, 1)
        self.tip = self.tip + MAX_DELTA * action[:3]
        self.gripper = angle_of(action[3])
        # The jaws close on the can when it is beside the pads and the gripper is nearly shut.
        if not self.held and self.gripper < 0.95:
            beside = np.linalg.norm(self.can[:2] - self.tip[:2] - HELD_OFFSET * CLOSING) < 0.012
            self.held = (
                beside
                and abs(self.can[2] - self.tip[2] + CONFIG.container.grasp_height - 0.061) < 0.012
            )
        if self.held and self.gripper > 0.95:
            self.held = False
            self.can[2] = GOAL.rest_z + CONFIG.container.height / 2
        if self.held:
            self.can = np.array(
                [
                    *(self.tip[:2] + HELD_OFFSET * CLOSING),
                    self.tip[2] - CONFIG.container.grasp_height + CONFIG.container.height / 2,
                ]
            )
        self.previous = action


def run(controller, world, steps=700):
    controller.reset(world.observe())
    phases = []
    for _ in range(steps):
        action = controller.act(world.observe())
        world.apply(action)
        if not phases or phases[-1] != controller.phase:
            phases.append(controller.phase)
        if controller.phase is Phase.DONE:
            break
    return phases


def test_the_phases_come_in_order_and_the_can_ends_on_the_goal():
    world = KinematicWorld()
    phases = run(ScriptedController(CONFIG), world)
    assert phases == [
        Phase.ABOVE,
        Phase.DESCEND,
        Phase.CLOSE,
        Phase.LIFT,
        Phase.CARRY,
        Phase.LOWER,
        Phase.RELEASE,
        Phase.RETREAT,
        Phase.DONE,
    ]
    assert world.can[:2] == pytest.approx([GOAL.center_x, GOAL.center_y], abs=0.006)
    assert world.can[2] == pytest.approx(GOAL.rest_z + 0.061, abs=0.006)


def test_the_first_action_heads_for_the_point_beside_and_above_the_can():
    world = KinematicWorld()
    controller = ScriptedController(CONFIG)
    controller.reset(world.observe())
    action = controller.act(world.observe())
    tcp_x, tcp_y = CAN[:2] - (0.033 + 0.002 - 0.0079) * CLOSING
    wanted = (
        np.array([tcp_x, tcp_y, 0.10 + 0.04]) - world.tip
    )  # above the grasp point by the clearance
    assert controller.phase is Phase.ABOVE
    assert np.sign(action[:3]).tolist() == np.sign(wanted).tolist()
    assert np.all(np.abs(action[:3]) <= 1)


def test_squeezing_ramps_the_gripper_down_at_the_configured_rate_and_then_waits():
    world = KinematicWorld()
    controller = ScriptedController(CONFIG)
    controller.reset(world.observe())
    # Walk to the grasp point first.
    while controller.phase is not Phase.CLOSE:
        world.apply(controller.act(world.observe()))
    angles = []
    steps = 0
    while controller.phase is Phase.CLOSE:
        action = controller.act(world.observe())
        world.apply(action)
        angles.append(angle_of(action[3]))
        steps += 1
    drops = -np.diff(
        angles[: int((CONFIG.grasp.open_angle - CONFIG.grasp.close_target) / 0.02) - 2]
    )
    assert drops == pytest.approx(CONFIG.grasp.close_rate, abs=1e-9)
    assert min(angles) == pytest.approx(CONFIG.grasp.close_target, abs=1e-9)
    # 70 steps to ramp from 1.3 to -0.1 at 0.02 per step, then 20 steps of waiting.
    assert steps == pytest.approx(90, abs=2)


def test_without_squeeze_the_can_is_never_lifted():
    world = KinematicWorld()
    run(ScriptedController(CONFIG, squeeze=False), world)
    assert world.can == pytest.approx(CAN)


def test_when_done_the_controller_stands_still_with_the_gripper_open():
    world = KinematicWorld()
    controller = ScriptedController(CONFIG)
    run(controller, world)
    action = controller.act(world.observe())
    assert action[:3] == pytest.approx(np.zeros(3))
    assert angle_of(action[3]) == pytest.approx(CONFIG.grasp.open_angle, abs=1e-6)


def test_resetting_starts_the_sequence_again():
    world = KinematicWorld()
    controller = ScriptedController(CONFIG)
    run(controller, world)
    controller.reset(KinematicWorld().observe())
    assert controller.phase is Phase.ABOVE
