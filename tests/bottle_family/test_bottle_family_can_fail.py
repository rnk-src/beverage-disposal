"""The family tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

FAMILY_TESTS = "tests/bottle_family/test_bottle_family.py"
CONFIG_TESTS = "tests/scene/test_scene_config.py"
SCENE_TESTS = "tests/bottle_family/test_bottle_scene.py"
FAMILY = "core/bottle_family.py"
CONFIG = "core/config.py"


def ignore_the_overfill(package):
    edit(package, FAMILY, "wanted = spec.nominal_ml * overfill", "wanted = spec.nominal_ml")


def accept_any_body_height(package):
    edit(
        package,
        FAMILY,
        "if not spec.body_height.lo <= height <= spec.body_height.hi:",
        "if False:",
    )


def call_every_size_supported(package):
    edit(package, FAMILY, 'size.gate.status == "passed"', "True")


def read_the_material_in_centimetres(package):
    edit(package, FAMILY, "MM = 1e-3", "MM = 1e-2")


def always_take_the_first_base_style(package):
    edit(
        package,
        FAMILY,
        "spec.base_styles[int(rng.integers(len(spec.base_styles)))]",
        "spec.base_styles[0]",
    )


def always_fit_the_cap(package):
    edit(package, FAMILY, "if rng.random() < CAP_PRESENT_P", "if True")


def skip_the_check_of_each_list_element(package):
    edit(
        package,
        CONFIG,
        'tuple(_convert(element, item, f"{where}[{i}]") for i, item in enumerate(value))',
        "tuple(value)",
    )


def grasp_the_neck_at_the_body_width(package):
    edit(
        package,
        FAMILY,
        "grasp_diameter = shape.neck_diameter",
        "grasp_diameter = shape.body_diameter",
    )


def forget_the_ring_drop(package):
    edit(
        package,
        FAMILY,
        "- shape.support_ring_drop - size.grasp_below_ring",
        "- size.grasp_below_ring",
    )


def allow_a_neck_grasp_without_a_ring(package):
    edit(package, FAMILY, "if shape.support_ring_diameter <= 0:", "if False:")


def ignore_the_drinks_density(package):
    edit(package, FAMILY, "fill_ml=liquid_ml * density_g_ml,", "fill_ml=liquid_ml,")


def keep_the_scenes_pitch(package):
    edit(package, FAMILY, "pitch_deg=size.grasp_pitch_deg", "pitch_deg=scene.grasp.pitch_deg")


FAULTS = [
    (
        ignore_the_overfill,
        FAMILY_TESTS,
        "test_the_brim_capacity_is_the_nominal_size_times_an_overfill_inside_its_range[pet_250ml]",
    ),
    (
        accept_any_body_height,
        FAMILY_TESTS,
        "test_a_size_whose_body_height_range_cannot_hold_its_capacity_gives_up_and_names_it",
    ),
    (call_every_size_supported, FAMILY_TESTS, "test_only_sizes_that_passed_the_gate_are_supported"),
    (
        read_the_material_in_centimetres,
        FAMILY_TESTS,
        "test_walls_finish_and_density_come_from_the_material_in_metres_and_kilograms",
    ),
    (
        always_take_the_first_base_style,
        FAMILY_TESTS,
        "test_the_base_style_comes_from_the_sizes_own_list",
    ),
    (
        always_fit_the_cap,
        FAMILY_TESTS,
        "test_a_bottle_may_lack_its_cap_or_its_label_and_often_has_both",
    ),
    (
        grasp_the_neck_at_the_body_width,
        SCENE_TESTS,
        "test_a_neck_grasp_takes_the_bottle_below_its_ring_at_the_neck_width",
    ),
    (
        forget_the_ring_drop,
        SCENE_TESTS,
        "test_a_neck_grasp_takes_the_bottle_below_its_ring_at_the_neck_width",
    ),
    (
        allow_a_neck_grasp_without_a_ring,
        SCENE_TESTS,
        "test_a_neck_grasp_needs_a_ring_to_stay_below",
    ),
    (
        ignore_the_drinks_density,
        SCENE_TESTS,
        "test_the_scene_for_a_bottle_holds_it_with_the_liquid_as_millilitres_of_water",
    ),
    (
        keep_the_scenes_pitch,
        SCENE_TESTS,
        "test_the_scene_for_a_bottle_holds_it_with_the_liquid_as_millilitres_of_water",
    ),
    (
        skip_the_check_of_each_list_element,
        CONFIG_TESTS,
        "test_a_tuple_field_refuses_a_non_list_or_a_wrong_element[value1]",
    ),
]


@pytest.mark.parametrize(
    ("fault", "tests", "failing_test"), FAULTS, ids=lambda value: getattr(value, "__name__", None)
)
def test_the_family_tests_fail_on_a_mutated_package(tmp_path, fault, tests, failing_test):
    result = run_on_mutated_package(tmp_path, tests, fault)
    assert_the_test_failed(result, tests, failing_test)
