"""The probe tests are run against mutated copies of the package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

PROBE_TESTS = "tests/probes/test_probes.py"
MATERIAL_TESTS = "tests/materials/test_materials.py"
BOUNDS_TESTS = "tests/bounds/test_bounds.py"
ZONE_TESTS = "tests/zones/test_zones.py"
POLICY_TESTS = "tests/policy/test_policy.py"
EVAL_TESTS = "tests/policy/test_emptiness_eval.py"
PROBES = "core/probes.py"
MATERIALS = "core/materials.py"
ZONES = "core/zones.py"
POLICY = "core/policy.py"
EVAL = "core/emptiness_eval.py"


def build_the_wall_reading_inside_out(package):
    edit(
        package,
        PROBES,
        "wall = (estimate / (1 + h), estimate / (1 - h))",
        "wall = (estimate * (1 - h), estimate * (1 + h))",
    )


def ignore_the_liquid_gain(package):
    edit(package, PROBES, "liquid_ml * liquid_gain * (", "liquid_ml * (")


def leave_the_additive_error_out_of_the_liquids_high_end(package):
    edit(package, PROBES, "max(estimate + a, 0.0) / (1 - r)", "max(estimate, 0.0) / (1 - r)")


def take_the_lower_of_the_two_low_ends_of_a_wall(package):
    edit(
        package,
        MATERIALS,
        "lo = max(material.t_side_mm.lo, side_wall_mm[0])",
        "lo = min(material.t_side_mm.lo, side_wall_mm[0])",
    )


def ignore_the_measured_wall(package):
    edit(
        package,
        MATERIALS,
        "return dataclasses.replace(material, t_side_mm=narrowed)",
        "return material",
    )


def return_the_union_of_two_liquid_intervals(package):
    edit(package, ZONES, "lo = max(first[0], second[0])", "lo = min(first[0], second[0])")


def compare_the_level_with_the_probes_low_end(package):
    edit(
        package,
        POLICY,
        "level_ml[0] > reading[1] + self.config.level_slack_ml",
        "level_ml[0] > reading[0] + self.config.level_slack_ml",
    )


def drop_a_reading_at_exactly_the_reach(package):
    edit(
        package,
        POLICY,
        "v_bounds[1] > self.config.probe_reach_ml",
        "v_bounds[1] >= self.config.probe_reach_ml",
    )


def bin_on_the_probe_when_it_contradicts_the_weight(package):
    edit(
        package,
        POLICY,
        "else intersect_ml(v_bounds, reading)",
        "else (intersect_ml(v_bounds, reading) or reading)",
    )


def draw_the_probes_from_the_bottles_generator(package):
    edit(package, EVAL, "probe_rng=probe_rng,", "probe_rng=rng,")


def ignore_the_thick_drinks_density(package):
    edit(
        package,
        EVAL,
        "low, high = density_g_per_ml or (",
        "low, high = None or (",
    )


def count_thick_drinks_among_the_sound_bottles(package):
    edit(package, EVAL, "        and cell.density_g_per_ml is None\n", "")


FAULTS = [
    (
        build_the_wall_reading_inside_out,
        PROBE_TESTS,
        "test_the_wall_reading_is_the_estimate_over_one_plus_and_minus_the_half_width",
    ),
    (
        ignore_the_liquid_gain,
        PROBE_TESTS,
        "test_a_low_reading_probe_puts_the_truth_above_its_reading",
    ),
    (
        leave_the_additive_error_out_of_the_liquids_high_end,
        PROBE_TESTS,
        "test_the_liquid_reading_by_hand",
    ),
    (
        take_the_lower_of_the_two_low_ends_of_a_wall,
        MATERIAL_TESTS,
        "test_a_measured_wall_inside_the_range_replaces_it",
    ),
    (
        ignore_the_measured_wall,
        BOUNDS_TESTS,
        "test_a_2000_ml_bottle_with_a_measured_wall_worked_by_hand",
    ),
    (
        return_the_union_of_two_liquid_intervals,
        ZONE_TESTS,
        "test_two_liquid_intervals_overlap_by_hand",
    ),
    (
        compare_the_level_with_the_probes_low_end,
        POLICY_TESTS,
        "test_the_camera_seeing_more_than_the_liquid_probe_is_a_veto",
    ),
    (
        drop_a_reading_at_exactly_the_reach,
        POLICY_TESTS,
        "test_a_liquid_reading_is_dropped_when_the_weight_allows_more_than_the_reach",
    ),
    (
        bin_on_the_probe_when_it_contradicts_the_weight,
        POLICY_TESTS,
        "test_a_liquid_reading_the_weight_contradicts_is_a_veto",
    ),
    (
        draw_the_probes_from_the_bottles_generator,
        EVAL_TESTS,
        "test_a_grid_with_probes_that_never_read_is_the_grid_without_them",
    ),
    (
        ignore_the_thick_drinks_density,
        EVAL_TESTS,
        "test_a_thick_drink_weighs_by_its_own_density",
    ),
    (
        count_thick_drinks_among_the_sound_bottles,
        EVAL_TESTS,
        "test_a_thick_drink_cell_is_counted_apart_from_the_sound_and_the_under_reading_bottles",
    ),
]


@pytest.mark.medium
@pytest.mark.parametrize(
    ("fault", "tests", "failing_test"), FAULTS, ids=lambda value: getattr(value, "__name__", None)
)
def test_the_probe_tests_fail_on_a_mutated_package(tmp_path, fault, tests, failing_test):
    result = run_on_mutated_package(tmp_path, tests, fault, only=failing_test)
    assert_the_test_failed(result, tests, failing_test)
