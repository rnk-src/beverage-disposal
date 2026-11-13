"""The scripted-controller tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

OBSERVATION_TESTS = "tests/scripted/test_observation.py"
GOAL_TESTS = "tests/scripted/test_goals.py"
CONTROLLER_TESTS = "tests/scripted/test_scripted_controller.py"
ROLLOUT_TESTS = "tests/scripted/test_scripted_rollout.py"
JUDGING_TESTS = "tests/scripted/test_judging.py"
INTERFACES = "core/interfaces.py"
BASELINES = "core/baselines.py"
LAYOUT = "core/layout.py"
SCRIPTED = "core/scripted.py"
ROLLOUT = "sim/rollout.py"


def double_the_noise(package):
    edit(
        package,
        INTERFACES,
        "rng.normal(0.0, self.sigma, size=3)",
        "rng.normal(0.0, 2 * self.sigma, size=3)",
    )


def accept_negative_noise(package):
    edit(package, INTERFACES, "        if self.sigma < 0:", "        if False:")


def let_the_random_policy_overreach(package):
    edit(package, BASELINES, "uniform(-1.0, 1.0, size=4)", "uniform(-2.0, 2.0, size=4)")


def give_the_bin_goal_the_outer_edge(package):
    edit(
        package,
        LAYOUT,
        "            bin_.inner_x / 2,\n            bin_.inner_y / 2,",
        "            bin_.inner_x / 2 + bin_.wall,\n            bin_.inner_y / 2 + bin_.wall,",
    )


def release_the_can_at_the_wall_top(package):
    edit(package, LAYOUT, "bin_.height + config.reach.release_clearance", "bin_.height")


def never_ramp_the_gripper_shut(package):
    edit(
        package,
        SCRIPTED,
        "            if self._squeeze:\n                self._angle = max(",
        "            if False:\n                self._angle = max(",
    )


def skip_the_wait_after_closing(package):
    edit(
        package,
        SCRIPTED,
        "self._ramp >= self._ramp_steps + SETTLE_STEPS",
        "self._ramp >= self._ramp_steps",
    )


def carry_the_tip_not_the_can_to_the_goal(package):
    edit(package, SCRIPTED, "goal[:2] - container[:2]", "goal[:2]")


def never_open_the_gripper_at_the_goal(package):
    edit(
        package,
        SCRIPTED,
        "self._angle = min(grasp.open_angle, self._angle + RELEASE_RATE)",
        "self._angle = self._angle",
    )


def accept_any_place_as_inside_the_goal(package):
    edit(
        package,
        "core/task_tests.py",
        "    if not goal.holds(position[0], position[1], radius):",
        "    if False:",
    )


def spawn_every_seed_at_the_same_place(package):
    edit(
        package,
        ROLLOUT,
        "sample_spawn(config, np.random.default_rng(seed))",
        "sample_spawn(config, np.random.default_rng(0))",
    )


def ignore_the_noise_model(package):
    edit(
        package,
        ROLLOUT,
        "return observation if noise is None else noise.apply(observation, noise_rng)",
        "return observation",
    )


FAULTS = [
    (
        OBSERVATION_TESTS,
        double_the_noise,
        "test_noise_is_gaussian_with_the_stated_spread_and_repeatable_by_seed",
    ),
    (OBSERVATION_TESTS, accept_negative_noise, "test_a_negative_noise_spread_is_rejected"),
    (
        OBSERVATION_TESTS,
        let_the_random_policy_overreach,
        "test_a_random_controller_acts_with_four_numbers_in_range_and_is_a_controller",
    ),
    (
        GOAL_TESTS,
        give_the_bin_goal_the_outer_edge,
        "test_the_bin_goal_is_its_free_floor_not_its_outer_edge",
    ),
    (
        GOAL_TESTS,
        release_the_can_at_the_wall_top,
        "test_there_are_three_goals_with_their_release_and_rest_heights",
    ),
    (
        CONTROLLER_TESTS,
        never_ramp_the_gripper_shut,
        "test_squeezing_ramps_the_gripper_down_at_the_configured_rate_and_then_waits",
    ),
    (
        CONTROLLER_TESTS,
        skip_the_wait_after_closing,
        "test_squeezing_ramps_the_gripper_down_at_the_configured_rate_and_then_waits",
    ),
    (
        CONTROLLER_TESTS,
        carry_the_tip_not_the_can_to_the_goal,
        "test_the_phases_come_in_order_and_the_can_ends_on_the_goal",
    ),
    (
        CONTROLLER_TESTS,
        never_open_the_gripper_at_the_goal,
        "test_the_phases_come_in_order_and_the_can_ends_on_the_goal",
    ),
    (
        JUDGING_TESTS,
        accept_any_place_as_inside_the_goal,
        "test_a_can_just_inside_the_edge_is_a_success_and_just_outside_is_not",
    ),
    (
        ROLLOUT_TESTS,
        spawn_every_seed_at_the_same_place,
        "test_each_seed_spawns_the_can_somewhere_else",
    ),
    (
        ROLLOUT_TESTS,
        ignore_the_noise_model,
        "test_five_millimetres_of_perception_noise_breaks_the_unfiltered_script",
    ),
]


@pytest.mark.medium
@pytest.mark.parametrize(
    ("module", "fault", "failing_test"),
    FAULTS,
    ids=lambda value: getattr(value, "__name__", None) if callable(value) else None,
)
def test_the_scripted_tests_fail_on_a_mutated_package(tmp_path, module, fault, failing_test):
    result = run_on_mutated_package(tmp_path, module, fault)
    assert_the_test_failed(result, module, failing_test)
