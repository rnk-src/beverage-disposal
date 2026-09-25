"""The Gazebo tests can fail: each fault, applied alone to a copy, breaks the right tests."""

import pytest
from mutants import edit, run_on_damaged_copy

pytestmark = pytest.mark.medium

DIR = "tests/gazebo/"
PLATFORM = DIR + "test_platform.py"
FALL = DIR + "test_falling_body.py"
IMAGE = DIR + "test_image.py"
GROUND = DIR + "test_ground_contact.py"
INCLINE = DIR + "test_incline.py"
REPRO = DIR + "test_reproducibility.py"
COMPARE = DIR + "test_state_comparison.py"
NO_DAEMON = {"DOCKER_HOST": "unix:///nonexistent/docker.sock"}

SCRIPT = "docker/gazebo/step_world.py"
DOCKERFILE = "docker/gazebo/Dockerfile"
FALLING_WORLD = "docker/gazebo/worlds/falling_body.sdf"
GROUND_WORLD = "docker/gazebo/worlds/ball_on_ground.sdf"
SLIDING_WORLD = "docker/gazebo/worlds/block_on_incline.sdf"
STICKY_WORLD = "docker/gazebo/worlds/block_on_incline_sticky.sdf"

ITERATIONS = "test_gazebo_advances_exactly_the_requested_iterations"
DETERMINISM = "test_two_runs_give_identical_poses"
DROP_TESTS = {
    "test_the_ball_drops_by_the_discrete_free_fall_distance",
    "test_the_drop_is_within_two_percent_of_continuous_free_fall",
    "test_gazebo_and_mujoco_drop_the_same_distance",
}
ARCH = "test_the_package_is_installed_for_the_container_architecture"
REST = "test_the_ball_comes_to_rest_with_its_lowest_point_on_the_ground"
REST_AGREE = "test_gazebo_and_mujoco_rest_at_the_same_height"
STATIC = "test_the_readout_lists_static_models_too"
SLIDE = "test_the_sliding_distance_matches_the_coulomb_formula"
SLIDE_AGREE = "test_gazebo_and_mujoco_slide_the_same_distance"
HOLDS = "test_gazebo_holds_the_block_still_when_friction_exceeds_the_slope"
CREEP = "test_known_gap_mujoco_creeps_where_gazebo_holds"
COMBINE = "test_known_gap_the_engines_combine_two_friction_coefficients_differently"
MIXED_WORLD = "docker/gazebo/worlds/block_on_incline_mixed.sdf"


def fault(selection, path, old, new, fails, may_also_fail=(), env=None, everywhere=False):
    return {
        "selection": selection,
        "edit": (path, old, new, everywhere),
        "fails": set(fails),
        "may_also_fail": set(may_also_fail),
        "env": env,
    }


FAULTS = {
    # Dockerfile and platform
    "dockerfile-step-fails": fault(
        IMAGE,
        DOCKERFILE,
        "RUN apt-get update",
        "RUN false\nRUN apt-get update",
        {"test_the_image_builds_from_the_dockerfile"},
    ),
    "gz-ros2-control-not-installed": fault(
        PLATFORM,
        DOCKERFILE,
        "        ros-jazzy-gz-ros2-control \\\n",
        "",
        {f"{ARCH}[ros-jazzy-gz-ros2-control]"},
    ),
    # Worlds
    "gravity-is-wrong": fault(
        FALL,
        FALLING_WORLD,
        "<gravity>0 0 -9.81</gravity>",
        "<gravity>0 0 -9.0</gravity>",
        DROP_TESTS,
    ),
    "step-size-is-wrong": fault(
        FALL,
        FALLING_WORLD,
        "<max_step_size>0.001</max_step_size>",
        "<max_step_size>0.002</max_step_size>",
        DROP_TESTS | {ITERATIONS},
    ),
    # The step script
    "step-request-unpauses-the-world": fault(
        FALL,
        SCRIPT,
        "request.pause = True",
        "request.pause = False",
        DROP_TESTS | {ITERATIONS},
        may_also_fail={DETERMINISM},
    ),
    "sim-time-scales-nanoseconds-wrongly": fault(
        FALL, SCRIPT, "stamp.nsec * 1e-9", "stamp.nsec * 1e-6", {ITERATIONS}
    ),
    "output-has-a-wall-clock-field": fault(
        FALL,
        SCRIPT,
        '"poses": model_poses(node, world),',
        '"poses": model_poses(node, world),\n            "wall": time.time(),',
        {DETERMINISM},
    ),
    "iterations-reported-off-by-one": fault(
        FALL,
        SCRIPT,
        '"iterations": state.stats.iterations,',
        '"iterations": state.stats.iterations + 1,',
        {ITERATIONS, "test_zero_steps_read_the_released_pose_without_stepping"},
    ),
    "readout-drops-the-static-ground": fault(
        GROUND,
        SCRIPT,
        "    return poses\n",
        '    return {k: v for k, v in poses.items() if k != "ground"}\n',
        {STATIC},
    ),
    # Ground and incline scenes
    "ball-radius-is-wrong": fault(
        GROUND, GROUND_WORLD, "<radius>0.1</radius>", "<radius>0.15</radius>", {REST, REST_AGREE}
    ),
    "ground-is-lower-than-expected": fault(
        GROUND,
        GROUND_WORLD,
        "<pose>0 0 -0.1 0 0 0</pose>",
        "<pose>0 0 -0.3 0 0 0</pose>",
        {STATIC, REST, REST_AGREE},
    ),
    "incline-gravity-is-wrong": fault(
        INCLINE,
        SLIDING_WORLD,
        "<gravity>0 0 -9.81</gravity>",
        "<gravity>0 0 -5.0</gravity>",
        {SLIDE, SLIDE_AGREE},
    ),
    "sticky-world-is-not-sticky": fault(
        INCLINE, STICKY_WORLD, "0.7", "0.3", {HOLDS}, everywhere=True
    ),
    "mixed-world-has-equal-friction": fault(
        INCLINE, MIXED_WORLD, "<mu>0.1</mu><mu2>0.1</mu2>", "<mu>0.3</mu><mu2>0.3</mu2>", {COMBINE}
    ),
    # Reproducibility
    "base-image-is-not-pinned": fault(
        REPRO + "::test_the_base_image_is_pinned_by_digest",
        DOCKERFILE,
        "@sha256:",
        "",
        {"test_the_base_image_is_pinned_by_digest"},
    ),
    "a-recorded-version-is-changed": fault(
        REPRO,
        "docker/gazebo/apt-versions.txt",
        "ros-jazzy-ros-gz-sim 1.0.24",
        "ros-jazzy-ros-gz-sim 1.0.23",
        {"test_the_installed_packages_are_exactly_the_recorded_versions"},
    ),
    "an-extra-package-is-installed": fault(
        REPRO,
        DOCKERFILE,
        "        ros-jazzy-gz-ros2-control \\\n",
        "        ros-jazzy-gz-ros2-control \\\n        nano \\\n",
        {"test_the_installed_packages_are_exactly_the_recorded_versions"},
    ),
    # State comparison
    "compare-positions-exactly": fault(
        COMPARE,
        "tests/gazebo/states.py",
        'first["poses"][name] == pytest.approx(second["poses"][name])',
        'first["poses"][name] == second["poses"][name]',
        {"test_a_last_bit_difference_in_a_position_is_the_same"},
    ),
    "compare-positions-too-loosely": fault(
        COMPARE,
        "tests/gazebo/states.py",
        'pytest.approx(second["poses"][name])',
        'pytest.approx(second["poses"][name], rel=1e-2)',
        {"test_a_visible_difference_in_a_position_is_not_the_same"},
    ),
    "compare-iterations-approximately": fault(
        COMPARE,
        "tests/gazebo/states.py",
        'first["iterations"] == second["iterations"]',
        'first["iterations"] == pytest.approx(second["iterations"], rel=0.05)',
        {"test_a_different_iteration_count_is_not_the_same_however_close_the_times"},
    ),
    "compare-ignores-extra-top-level-fields": fault(
        COMPARE,
        "tests/gazebo/states.py",
        'first.keys() != second.keys() or first["poses"].keys()',
        'first["poses"].keys()',
        {"test_an_extra_field_is_not_the_same"},
    ),
    "compare-ignores-fields-it-does-not-know": fault(
        COMPARE,
        "tests/gazebo/states.py",
        "all(first[key] == second[key] for key in others)",
        "True",
        {"test_a_field_that_differs_between_two_runs_is_not_the_same"},
    ),
    "compare-ignores-missing-models": fault(
        COMPARE,
        "tests/gazebo/states.py",
        ' or first["poses"].keys() != second["poses"].keys()',
        "",
        {"test_a_missing_model_is_not_the_same"},
    ),
    # Expectations inside the tests
    "test-expects-the-wrong-ubuntu": fault(
        PLATFORM, PLATFORM, '"24.04"', '"22.04"', {"test_the_image_is_ubuntu_2404"}
    ),
    "test-expects-the-wrong-python": fault(
        PLATFORM, PLATFORM, '"Python 3.12."', '"Python 3.11."', {"test_the_system_python_is_312"}
    ),
    "test-expects-the-wrong-distribution": fault(
        PLATFORM, PLATFORM, '== "jazzy"', '== "humble"', {"test_the_ros_distribution_is_jazzy"}
    ),
    "test-expects-the-wrong-gazebo": fault(
        PLATFORM, PLATFORM, 'startswith("8.")', 'startswith("7.")', {"test_gazebo_is_harmonic"}
    ),
    "test-accepts-only-architecture-all": fault(
        PLATFORM,
        PLATFORM,
        'assert installed == in_image(built_image, "dpkg", "--print-architecture")',
        'assert installed == "all"',
        {
            f"{ARCH}[{p}]"
            for p in (
                "ros-jazzy-ros-gz-sim",
                "ros-jazzy-ros-gz-bridge",
                "ros-jazzy-gz-ros2-control",
            )
        },
    ),
    "test-uses-the-continuous-formula-for-the-discrete-drop": fault(
        FALL,
        FALL,
        "STEPS * (STEPS + 1) / 2",
        "STEPS * STEPS / 2",
        {"test_the_ball_drops_by_the_discrete_free_fall_distance"},
    ),
    "test-expects-the-wrong-rest-height": fault(
        GROUND, GROUND, "RADIUS = 0.1", "RADIUS = 0.2", {REST}
    ),
    "test-uses-the-wrong-coulomb-formula": fault(
        INCLINE,
        INCLINE,
        "math.sin(INCLINE) - friction * math.cos(INCLINE)",
        "math.sin(INCLINE) - friction",
        # coulomb_distance is shared, so the mixed-friction test breaks too.
        {SLIDE, COMBINE},
    ),
    "test-swaps-min-and-max-for-gazebo": fault(
        INCLINE,
        INCLINE,
        "coulomb_distance(min(slab, block), 1.0)",
        "coulomb_distance(max(slab, block), 1.0)",
        {COMBINE},
    ),
    "test-moves-the-creep-band-above-the-measurement": fault(
        INCLINE, INCLINE, "0.5e-3 <", "10e-3 <", {CREEP}
    ),
    "conftest-no-longer-skips-without-a-daemon": fault(
        IMAGE,
        DIR + "conftest.py",
        'pytest.skip("the Docker daemon is not running")',
        "pass",
        {"test_the_image_builds_from_the_dockerfile"},
        env=NO_DAEMON,
    ),
}


@pytest.mark.parametrize("name", FAULTS)
def test_the_fault_breaks_exactly_the_tests_that_should_catch_it(tmp_path, name):
    spec = FAULTS[name]
    path, old, new, everywhere = spec["edit"]
    _, found = run_on_damaged_copy(
        tmp_path,
        lambda copy: edit(copy, path, old, new, everywhere),
        spec["selection"],
        env=spec["env"],
    )
    failed = set(found["FAILED"])
    assert spec["fails"] <= failed <= spec["fails"] | spec["may_also_fail"], found
