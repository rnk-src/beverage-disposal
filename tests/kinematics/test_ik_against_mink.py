"""Our IK against mink, a MuJoCo-based IK library, solving the same problems (test oracle only).

mink runs MuJoCo's own kinematics and has its own joint-limit handling, so it can disagree with
our NumPy IK in ways our own FK-based checks cannot see (a shared mistake in the chain file, say).
"""

import mink
import numpy as np
import pytest
from support.so101 import MJCF, SO101

from beverage_disposal.core.ik import solve_ik
from beverage_disposal.core.kinematics import HOME, fk, pitch
from beverage_disposal.sim.arm import load_arm

pytestmark = pytest.mark.medium

DT = 0.01
MINK_ITERATIONS = 300


@pytest.fixture(scope="module")
def arm():
    return load_arm(SO101 / MJCF)


def solve_with_mink(model, start, target_position, target_rotation):
    """Drive the TCP site to the target with mink; the rotation is ignored when it is None."""
    q = np.zeros(model.nq)
    q[:5] = start
    configuration = mink.Configuration(model, q)
    task = mink.FrameTask(
        "tcp",
        "site",
        position_cost=1.0,
        orientation_cost=0.0 if target_rotation is None else 1.0,
        gain=0.3,
        lm_damping=1e-3,
    )
    rotation = np.eye(3) if target_rotation is None else target_rotation
    task.set_target(
        mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rotation), target_position)
    )
    limits = [mink.ConfigurationLimit(model)]
    for _ in range(MINK_ITERATIONS):
        velocity = mink.solve_ik(configuration, [task], DT, "daqp", damping=1e-6, limits=limits)
        configuration.integrate_inplace(velocity, DT)
    tip = configuration.data.site("tcp").xpos.copy()
    return configuration.q[:5].copy(), float(np.linalg.norm(target_position - tip))


def test_ik_and_mink_find_the_same_joint_angles_for_a_reachable_pose_near_home(arm):
    rng = np.random.default_rng(0)
    for _ in range(15):
        pose = HOME + np.append(rng.uniform(-0.4, 0.4, 4), 0.0)
        pose = np.clip(pose, *arm.jnt_range[:5].T + [[1e-5], [-1e-5]])
        pose_matrix = fk(pose)
        ours = solve_ik(pose_matrix[:3, 3], target_pitch=pitch(pose), start=HOME)
        mink_angles, mink_error = solve_with_mink(
            arm, HOME, pose_matrix[:3, 3], pose_matrix[:3, :3]
        )
        assert ours.converged
        assert mink_error < 1e-3
        np.testing.assert_allclose(ours.angles, mink_angles, rtol=0, atol=5e-3)


def test_ik_ends_no_further_from_a_target_out_of_reach_than_mink_and_both_respect_the_limits(arm):
    rng = np.random.default_rng(1)
    low, high = arm.jnt_range[:5].T
    for _ in range(15):
        direction = rng.normal(size=3)
        direction[0], direction[2] = abs(direction[0]), abs(direction[2])
        # Half a metre from the shoulder is a little beyond the arm's longest stretch (about 0.4 m).
        target = np.array([0.0388, 0.0, 0.1166]) + 0.5 * direction / np.linalg.norm(direction)
        ours = solve_ik(target, target_pitch=None, start=HOME)
        mink_angles, mink_error = solve_with_mink(arm, HOME, target, None)
        assert not ours.converged
        # mink can end in a worse spot (no step cap: it sometimes jumps to the wrong side of a
        # joint limit), so the claim is one-sided: ours is never meaningfully further away.
        assert ours.position_error <= mink_error + 5e-3
        for angles in (ours.angles, mink_angles):
            assert np.all(angles >= low - 1e-6) and np.all(angles <= high + 1e-6)
