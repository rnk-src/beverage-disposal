"""The layout tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

LAYOUT_TESTS = "tests/scene/test_layout.py"
REACH_TESTS = "tests/scene/test_reach.py"
PHYSICS_TESTS = "tests/scene/test_layout_physics.py"
LAYOUT = "core/layout.py"
REACH = "core/reach.py"
SCENE = "sim/scene.py"


def leave_the_wall_out_of_the_bin_footprint(package):
    edit(package, LAYOUT, "bin_.inner_x / 2 + bin_.wall,", "bin_.inner_x / 2,")


def double_the_spawn_box(package):
    edit(package, LAYOUT, "config.spawn.half_size,", "2 * config.spawn.half_size,")


def always_spawn_at_the_corner(package):
    edit(package, LAYOUT, "float(rng.uniform(x_min, x_max))", "float(x_min)")


def forget_the_cans_radius_in_the_set_aside_test(package):
    edit(package, LAYOUT, "area.half_x - radius", "area.half_x")


def put_the_poses_on_the_table_not_at_the_grasp_height(package):
    edit(package, REACH, "(x, y, base + grasp)))", "(x, y, base)))")


def release_without_the_wall_height(package):
    edit(package, REACH, "config.bin.height + reach.release_clearance", "reach.release_clearance")


def call_every_pose_reachable(package):
    edit(
        package,
        REACH,
        "if solve_ik(np.array(position), np.radians(pitch_deg)).converged:",
        "if True:",
    )


def lower_the_bin_walls(package):
    edit(package, SCENE, "middle = bin_.height / 2", "middle = bin_.height / 4")


def thin_the_bin_floor(package):
    edit(package, SCENE, "bin_.floor / 2", "bin_.floor / 4")


def let_the_marker_collide(package):
    edit(package, SCENE, "marker.contype = 0", "marker.contype = 1")


def sink_the_pad(package):
    edit(
        package,
        SCENE,
        "[pad.half_x, pad.half_y, pad.height / 2]",
        "[pad.half_x, pad.half_y, 0.005]",
    )


def shift_the_bin_walls(package):
    edit(package, SCENE, "bin_.center_y + sign", "bin_.center_y + 0.5 * sign")


LAYOUT_FAULTS = [
    (
        leave_the_wall_out_of_the_bin_footprint,
        "test_the_footprints_are_the_outer_edges_of_what_the_file_describes",
    ),
    (double_the_spawn_box, "test_spawn_positions_stay_in_the_box_and_use_all_of_it"),
    (always_spawn_at_the_corner, "test_spawn_positions_stay_in_the_box_and_use_all_of_it"),
    (
        forget_the_cans_radius_in_the_set_aside_test,
        "test_a_can_is_in_the_set_aside_area_only_when_its_whole_footprint_is",
    ),
]

REACH_FAULTS = [
    (
        put_the_poses_on_the_table_not_at_the_grasp_height,
        "test_the_poses_are_where_the_layout_and_the_grasp_height_put_them",
    ),
    (
        release_without_the_wall_height,
        "test_the_poses_are_where_the_layout_and_the_grasp_height_put_them",
    ),
    (call_every_pose_reachable, "test_a_bin_placed_out_of_reach_is_reported_by_name"),
]

PHYSICS_FAULTS = [
    (
        lower_the_bin_walls,
        "test_the_bin_walls_stand_at_the_configured_height_around_the_configured_centre",
    ),
    (thin_the_bin_floor, "test_a_can_dropped_over_the_bin_ends_up_on_the_bin_floor"),
    (let_the_marker_collide, "test_the_set_aside_marker_is_painted_on_and_does_not_hold_a_can_up"),
    (sink_the_pad, "test_a_can_on_the_pad_rests_on_the_pad_top"),
    (shift_the_bin_walls, "test_moving_the_bin_in_the_config_moves_its_walls"),
]


@pytest.mark.parametrize(
    ("module", "fault", "failing_test"),
    [(LAYOUT_TESTS, f, t) for f, t in LAYOUT_FAULTS]
    + [(REACH_TESTS, f, t) for f, t in REACH_FAULTS]
    + [(PHYSICS_TESTS, f, t) for f, t in PHYSICS_FAULTS],
    ids=lambda value: getattr(value, "__name__", None) if callable(value) else None,
)
def test_the_layout_tests_fail_on_a_mutated_package(tmp_path, module, fault, failing_test):
    result = run_on_mutated_package(tmp_path, module, fault)
    assert_the_test_failed(result, module, failing_test)
