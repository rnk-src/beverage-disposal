"""The bottle tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

TESTS = "tests/bottles/test_bottles.py"
BOTTLES = "core/bottles.py"


def use_the_wall_thickness_for_the_base(package):
    edit(package, BOTTLES, "found.base * shape.t_base", "found.base * shape.t_side")


def forget_the_shoulders_slant(package):
    edit(
        package, BOTTLES, "math.hypot(shape.shoulder_height, big - small)", "shape.shoulder_height"
    )


def drop_a_term_of_the_frustum(package):
    edit(package, BOTTLES, "(big**2 + big * radius + radius**2)", "(big**2 + radius**2)")


def start_the_liquid_at_the_outside_of_the_base(package):
    edit(package, BOTTLES, "total_height_m(shape)) - floor", "total_height_m(shape))")


def search_the_level_the_wrong_way(package):
    edit(package, BOTTLES, "< volume_ml:", "> volume_ml:")


def allow_a_neck_wider_than_the_body(package):
    edit(package, BOTTLES, "if self.neck_diameter > self.body_diameter:", "if False:")


def leave_the_cap_out_of_the_empty_mass(package):
    edit(
        package,
        BOTTLES,
        "shape.finish_mass + shape.cap_mass + shape.label_mass",
        "shape.finish_mass",
    )


def use_a_wrong_frustum_centroid(package):
    edit(package, BOTTLES, "3 * radius**2", "radius**2")


def let_the_ring_be_as_wide_as_the_neck(package):
    edit(
        package,
        BOTTLES,
        "self.neck_diameter < self.support_ring_diameter",
        "self.neck_diameter <= self.support_ring_diameter",
    )


def measure_the_centroid_from_the_inside_floor(package):
    edit(package, BOTTLES, "moment + part * (shape.t_base + up / 2)", "moment + part * (up / 2)")


FAULTS = [
    (
        use_the_wall_thickness_for_the_base,
        "test_the_shell_mass_is_density_times_each_area_times_its_thickness",
    ),
    (
        forget_the_shoulders_slant,
        "test_the_areas_are_side_base_and_the_shoulders_slanted_surface",
    ),
    (
        drop_a_term_of_the_frustum,
        "test_a_shape_with_a_conical_shoulder_holds_body_plus_frustum_plus_neck",
    ),
    (
        start_the_liquid_at_the_outside_of_the_base,
        "test_the_volume_below_a_level_in_the_body_is_a_cylinder_above_the_floor",
    ),
    (
        search_the_level_the_wrong_way,
        "test_the_level_for_a_volume_inverts_the_volume_below_a_level",
    ),
    (
        allow_a_neck_wider_than_the_body,
        "test_an_impossible_shape_is_refused[change0-neck]",
    ),
    (
        use_a_wrong_frustum_centroid,
        "test_the_centroid_of_body_plus_whole_shoulder_is_the_volume_weighted_mean",
    ),
    (
        let_the_ring_be_as_wide_as_the_neck,
        "test_an_impossible_support_ring_is_refused[change0-ring]",
    ),
    (
        measure_the_centroid_from_the_inside_floor,
        "test_the_centroid_sits_on_the_raised_floor_inside_the_walls",
    ),
    (
        leave_the_cap_out_of_the_empty_mass,
        "test_the_empty_mass_adds_the_neck_cap_and_label_to_the_shell",
    ),
]


@pytest.mark.parametrize(
    ("fault", "failing_test"), FAULTS, ids=lambda value: getattr(value, "__name__", None)
)
def test_the_bottle_tests_fail_on_a_mutated_package(tmp_path, fault, failing_test):
    result = run_on_mutated_package(tmp_path, TESTS, fault)
    assert_the_test_failed(result, TESTS, failing_test)
