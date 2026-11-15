"""The bottle physics tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

TESTS = "tests/scene/test_bottle_physics.py"
BOTTLE = "sim/bottle.py"
SCENE = "sim/scene.py"


def put_the_liquid_at_the_origin(package):
    edit(package, BOTTLE, "pos=[0.0, 0.0, local(centroid)]", "pos=[0.0, 0.0, 0.0]")


def leave_the_cap_and_label_out_of_the_mass(package):
    edit(
        package,
        BOTTLE,
        "neck_mass = shape.finish_mass - ring_mass + shape.cap_mass + shape.label_mass",
        "neck_mass = shape.finish_mass - ring_mass",
    )


def count_the_liquid_twice(package):
    edit(
        package,
        BOTTLE,
        "mass = liquid_ml * density_g_ml * LIQUID_KG_PER_ML",
        "mass = 2 * liquid_ml * density_g_ml * LIQUID_KG_PER_ML",
    )


def forget_the_drinks_density_in_the_mass(package):
    edit(
        package,
        BOTTLE,
        "mass = liquid_ml * density_g_ml * LIQUID_KG_PER_ML",
        "mass = liquid_ml * LIQUID_KG_PER_ML",
    )


def find_the_centroid_of_the_mass_as_if_it_were_volume(package):
    edit(
        package,
        BOTTLE,
        "centroid = liquid_centroid_m(shape, liquid_ml)",
        "centroid = liquid_centroid_m(shape, liquid_ml * density_g_ml)",
    )


def put_the_ring_two_millimetres_low(package):
    edit(
        package,
        BOTTLE,
        "total - shape.support_ring_drop + RING_THICKNESS / 2",
        "total - shape.support_ring_drop - RING_THICKNESS / 2",
    )


def start_the_shoulder_at_the_base(package):
    edit(
        package,
        BOTTLE,
        "low, high = shape.body_height, shape.body_height + shape.shoulder_height",
        "low, high = 0.0, shape.shoulder_height",
    )


def let_the_liquid_level_collide(package):
    edit(package, BOTTLE, "contype=0,", "contype=1,")


def ignore_the_bottle_in_the_scene(package):
    edit(package, SCENE, "if config.bottle is not None:", "if False:")


FAULTS = [
    (
        put_the_liquid_at_the_origin,
        "test_the_liquid_sits_at_its_closed_form_centroid[pet_250ml]",
    ),
    (
        leave_the_cap_and_label_out_of_the_mass,
        "test_the_model_weighs_the_empty_bottle_plus_the_liquid[empty-pet_250ml]",
    ),
    (
        count_the_liquid_twice,
        "test_the_model_weighs_the_empty_bottle_plus_the_liquid[payload-pet_250ml]",
    ),
    (
        forget_the_drinks_density_in_the_mass,
        "test_a_denser_drink_weighs_more_but_stands_at_the_same_height[part]",
    ),
    (
        find_the_centroid_of_the_mass_as_if_it_were_volume,
        "test_a_denser_drink_weighs_more_but_stands_at_the_same_height[part]",
    ),
    (
        put_the_ring_two_millimetres_low,
        "test_the_ring_stands_at_its_drop_below_the_top_of_the_neck",
    ),
    (
        start_the_shoulder_at_the_base,
        "test_the_shoulder_spans_the_height_between_the_body_and_the_neck",
    ),
    (
        let_the_liquid_level_collide,
        "test_the_bottle_parts_collide_and_the_drawn_liquid_level_does_not",
    ),
    (
        ignore_the_bottle_in_the_scene,
        "test_the_ring_stands_at_its_drop_below_the_top_of_the_neck",
    ),
]


@pytest.mark.medium
@pytest.mark.parametrize(
    ("fault", "failing_test"), FAULTS, ids=lambda value: getattr(value, "__name__", None)
)
def test_the_bottle_physics_tests_fail_on_a_mutated_package(tmp_path, fault, failing_test):
    result = run_on_mutated_package(tmp_path, TESTS, fault, only=failing_test)
    assert_the_test_failed(result, TESTS, failing_test)
