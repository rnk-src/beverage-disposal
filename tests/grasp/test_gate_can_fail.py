"""The gate tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

GATE_TESTS = "tests/grasp/test_gate.py"
SCRIPT_TESTS = "tests/scripted/test_scripted_controller.py"
SCENE_TESTS = "tests/bottle_family/test_bottle_scene.py"
ROOM_TESTS = "tests/scripted/test_rollout_room.py"
REACH_TESTS = "tests/scene/test_bottle_reach.py"
GATE = "core/gate.py"
SCRIPT = "core/scripted.py"


def measure_the_slip_from_the_first_row(package):
    edit(package, GATE, "offset[last] - offset[grip_index]", "offset[last] - offset[0]")


def read_the_slip_after_the_release(package):
    edit(package, GATE, "last = release_index - 1", "last = len(trajectory.time) - 1")


def settle_at_the_first_entry_into_the_band(package):
    edit(
        package,
        GATE,
        "settled = int(outside[-1]) + 1",
        "settled = int(np.flatnonzero(np.abs(tilt - tilt[-1]) <= band_rad)[0])",
    )


def call_a_swing_that_is_still_going_settled(package):
    edit(package, GATE, "if len(tilt) - settled < MIN_QUIET_ROWS:", "if False:")


def accept_one_place_short(package):
    edit(
        package,
        GATE,
        "sum(row.places_upright) >= bar.required_places",
        "sum(row.places_upright) > bar.required_places - 2",
    )


def refuse_a_slip_at_the_limit(package):
    edit(package, GATE, "max(row.slips) <= bar.max_slip_m", "max(row.slips) < bar.max_slip_m")


def ignore_the_settle_time(package):
    edit(package, GATE, "and max(row.settle_times) <= bar.max_swing_settle_s", "")


def approach_at_the_grasp_clearance_whatever_the_height(package):
    edit(package, SCRIPT, "container.height + TOP_CLEARANCE,", "0.0,")


def log_every_step_as_a_phase(package):
    edit(package, SCRIPT, "if phase is not self.phase:", "if True:")


def keep_the_phase_log_across_resets(package):
    edit(
        package,
        SCRIPT,
        "self.phase_log = [(Phase.ABOVE, 0)]",
        "self.phase_log.append((Phase.ABOVE, 0))",
    )


def forget_the_rings_overhang(package):
    edit(package, "core/bottle_family.py", 'if size.grasp_mode == "neck":', "if False:")


def start_every_container_from_the_cans_ready_point(package):
    edit(
        package,
        "sim/rollout.py",
        "return (x, y, max(z, surface_z + approach_height_m(config)))",
        "return (x, y, z)",
    )


def carry_to_the_bin_at_the_release_height(package):
    edit(package, "core/reach.py", "max(lift_z, release_z + CARRY_CLEARANCE)", "release_z")


def stop_the_sweep_at_the_scenes_tilt_floor(package):
    edit(
        package,
        "core/reach.py",
        "pitches = range(90, -1, -PITCH_STEP_DEG)",
        "pitches = range(90, 59, -PITCH_STEP_DEG)",
    )


FAULTS = [
    (
        measure_the_slip_from_the_first_row,
        GATE_TESTS,
        "test_movement_before_the_grip_settled_does_not_count_as_slip",
    ),
    (
        read_the_slip_after_the_release,
        GATE_TESTS,
        "test_the_slip_is_read_on_the_last_row_before_the_release",
    ),
    (
        settle_at_the_first_entry_into_the_band,
        GATE_TESTS,
        "test_the_swing_settles_at_the_last_time_the_tilt_is_outside_the_band",
    ),
    (
        call_a_swing_that_is_still_going_settled,
        GATE_TESTS,
        "test_a_swing_that_is_still_going_at_the_end_never_settles",
    ),
    (accept_one_place_short, GATE_TESTS, "test_one_place_short_of_twelve_fails"),
    (
        refuse_a_slip_at_the_limit,
        GATE_TESTS,
        "test_a_slip_or_a_settle_time_at_the_limit_passes_and_just_over_fails",
    ),
    (
        ignore_the_settle_time,
        GATE_TESTS,
        "test_a_slip_or_a_settle_time_at_the_limit_passes_and_just_over_fails",
    ),
    (
        approach_at_the_grasp_clearance_whatever_the_height,
        SCRIPT_TESTS,
        "test_a_container_taller_than_the_approach_height_is_approached_over_its_top",
    ),
    (
        log_every_step_as_a_phase,
        SCRIPT_TESTS,
        "test_the_phase_log_holds_each_phase_once_with_the_step_it_began_on",
    ),
    (keep_the_phase_log_across_resets, SCRIPT_TESTS, "test_resetting_empties_the_phase_log"),
    (
        forget_the_rings_overhang,
        SCENE_TESTS,
        "test_a_neck_grasp_opens_the_approach_by_the_rings_overhang",
    ),
    (
        start_every_container_from_the_cans_ready_point,
        ROOM_TESTS,
        "test_a_tall_container_on_the_pad_is_started_from_above_its_approach_height",
    ),
    (
        carry_to_the_bin_at_the_release_height,
        REACH_TESTS,
        "test_the_poses_follow_the_leg_from_the_pad_at_the_containers_grasp_height",
    ),
    (
        stop_the_sweep_at_the_scenes_tilt_floor,
        REACH_TESTS,
        "test_the_sweep_goes_below_the_scenes_tilt_floor_down_to_level",
    ),
]


@pytest.mark.parametrize(
    ("fault", "tests", "failing_test"), FAULTS, ids=lambda value: getattr(value, "__name__", None)
)
def test_the_gate_tests_fail_on_a_mutated_package(tmp_path, fault, tests, failing_test):
    result = run_on_mutated_package(tmp_path, tests, fault)
    assert_the_test_failed(result, tests, failing_test)
