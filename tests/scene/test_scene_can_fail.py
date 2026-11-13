"""The scene tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

PHYSICS = "tests/scene/test_scene_physics.py"
CONFIG_TESTS = "tests/scene/test_scene_config.py"
SCENE = "sim/scene.py"
LOADER = "core/config.py"
CONTAINERS = "core/containers.py"


def stand_the_can_on_its_full_height(package):
    edit(
        package,
        SCENE,
        "size=[can.diameter / 2, can.height / 2, 0.0]",
        "size=[can.diameter / 2, can.height, 0.0]",
    )


def use_the_diameter_as_the_radius(package):
    edit(
        package,
        SCENE,
        "size=[can.diameter / 2, can.height / 2, 0.0]",
        "size=[can.diameter, can.height / 2, 0.0]",
    )


def forget_the_liquid(package):
    edit(package, SCENE, "mass=total_mass(can, config.fill_ml)", "mass=can.empty_mass")


def ignore_the_configured_position(package):
    edit(
        package,
        SCENE,
        "pos=[config.container_x, config.container_y, can.height / 2]",
        "pos=[0.3, 0.0, can.height / 2]",
    )


def sink_the_table_top_to_the_base(package):
    edit(
        package,
        SCENE,
        "pos=[table.center_x, 0.0, -table.thickness / 2]",
        "pos=[table.center_x, 0.0, -table.thickness]",
    )


def make_the_can_slippery(package):
    # MuJoCo takes the larger friction of the two touching geoms, so the table's has to drop too.
    for owner in ["can", "table"]:
        edit(
            package,
            SCENE,
            f"friction=[{owner}.friction, TORSIONAL_FRICTION, ROLLING_FRICTION]",
            "friction=[0.05, TORSIONAL_FRICTION, ROLLING_FRICTION]",
        )


def break_the_loop_for_scenes(package):
    edit(package, "sim/loop.py", "self._data.qpos[: ARM_JOINTS + 1] =", "self._data.qpos[:] =")


def skip_the_unknown_key_check(package):
    edit(package, LOADER, "    if unknown:", "    if False:")


def skip_the_missing_key_check(package):
    edit(package, LOADER, "    if missing:", "    if False:")


def accept_any_type(package):
    edit(
        package,
        LOADER,
        "    if not isinstance(value, hint) or isinstance(value, bool):",
        "    if False:",
    )


def accept_any_provenance_kind(package):
    edit(package, CONTAINERS, "        if self.kind not in PROVENANCE_KINDS:", "        if False:")


def take_ten_grams_per_millilitre(package):
    edit(package, CONTAINERS, "LIQUID_KILOGRAMS_PER_ML = 1e-3", "LIQUID_KILOGRAMS_PER_ML = 1e-2")


def accept_a_negative_fill(package):
    edit(package, CONTAINERS, "    if fill_ml < 0:", "    if False:")


PHYSICS_FAULTS = [
    (stand_the_can_on_its_full_height, "test_the_can_rests_at_half_its_height_on_the_table[empty]"),
    (
        use_the_diameter_as_the_radius,
        "test_the_can_in_the_model_has_the_configured_width_and_height",
    ),
    (forget_the_liquid, "test_the_model_mass_is_the_empty_mass_plus_a_gram_per_millilitre[355.0]"),
    (ignore_the_configured_position, "test_the_can_starts_where_the_config_puts_it"),
    (sink_the_table_top_to_the_base, "test_the_table_top_is_the_height_of_the_arms_base"),
    (
        make_the_can_slippery,
        "test_a_push_well_below_the_friction_limit_does_not_make_the_can_creep",
    ),
    (break_the_loop_for_scenes, "test_the_arm_loop_runs_in_the_scene_and_holds_its_pose"),
]

CONFIG_FAULTS = [
    (skip_the_unknown_key_check, "test_a_key_the_loader_does_not_know_is_rejected_and_named"),
    (skip_the_missing_key_check, "test_a_missing_key_is_rejected_and_named"),
    (accept_any_type, "test_a_value_of_the_wrong_type_is_rejected_and_named"),
    (accept_any_provenance_kind, "test_an_unknown_provenance_kind_is_rejected"),
    (take_ten_grams_per_millilitre, "test_the_contents_weigh_one_gram_per_millilitre"),
    (accept_a_negative_fill, "test_a_negative_fill_is_rejected"),
]


@pytest.mark.parametrize(
    ("fault", "failing_test"), PHYSICS_FAULTS, ids=[f.__name__ for f, _ in PHYSICS_FAULTS]
)
def test_the_scene_physics_tests_fail_on_a_mutated_package(tmp_path, fault, failing_test):
    result = run_on_mutated_package(tmp_path, PHYSICS, fault)
    assert_the_test_failed(result, PHYSICS, failing_test)


@pytest.mark.parametrize(
    ("fault", "failing_test"), CONFIG_FAULTS, ids=[f.__name__ for f, _ in CONFIG_FAULTS]
)
def test_the_scene_config_tests_fail_on_a_mutated_package(tmp_path, fault, failing_test):
    result = run_on_mutated_package(tmp_path, CONFIG_TESTS, fault)
    assert_the_test_failed(result, CONFIG_TESTS, failing_test)
