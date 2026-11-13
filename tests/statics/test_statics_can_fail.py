"""The statics tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

TESTS = "tests/statics/test_statics.py"
STATICS = "core/statics.py"


def drop_the_cosine_in_the_lever(package):
    edit(package, STATICS, "lever = x_centre * math.cos(tilt) -", "lever = x_centre -")


def let_the_rim_sit_at_the_axis(package):
    edit(
        package,
        STATICS,
        "rim_level = -radius * math.sin(tilt) + height * math.cos(tilt)",
        "rim_level = height * math.cos(tilt)",
    )


def ignore_the_density(package):
    edit(package, STATICS, "mass = retained * density * 1e3", "mass = retained * 1e3")


def lose_the_sign_of_the_tilt(package):
    edit(package, STATICS, "torque_nm=sign * mass * G * lever", "torque_nm=mass * G * lever")


def allow_a_flat_on_its_side_tilt(package):
    edit(package, STATICS, "    if abs(tilt) >= MAX_TILT:", "    if False:")


def allow_a_negative_fill(package):
    edit(package, STATICS, "    if fill_ml < 0:", "    if False:")


def try_no_parameter_errors_only(package):
    edit(
        package,
        STATICS,
        "itertools.product((-1.0, 1.0), repeat=5)",
        "itertools.product((0.0,), repeat=5)",
    )


def forget_to_keep_what_spills(package):
    edit(package, STATICS, "spilled_ml=(wanted - retained) * 1e6,", "spilled_ml=0.0,")


FAULTS = [
    (
        drop_the_cosine_in_the_lever,
        "test_a_tilted_container_matches_the_closed_form_for_a_level_that_misses_both_ends",
    ),
    (
        let_the_rim_sit_at_the_axis,
        "test_a_half_full_container_starts_to_spill_at_the_same_cutting_angle",
    ),
    (ignore_the_density, "test_the_liquid_torque_scales_with_the_density_and_nothing_else"),
    (lose_the_sign_of_the_tilt, "test_the_torque_flips_sign_with_the_direction_of_the_tilt"),
    (allow_a_flat_on_its_side_tilt, "test_a_tilt_of_ninety_degrees_or_more_is_refused"),
    (allow_a_negative_fill, "test_a_negative_fill_is_refused"),
    (
        try_no_parameter_errors_only,
        "test_the_density_error_alone_moves_the_torque_by_that_fraction_of_it",
    ),
    (
        forget_to_keep_what_spills,
        "test_the_liquid_that_stays_and_the_liquid_that_spills_add_up_to_what_was_poured",
    ),
]


@pytest.mark.parametrize(
    ("fault", "failing_test"), FAULTS, ids=[fault.__name__ for fault, _ in FAULTS]
)
def test_the_statics_tests_fail_on_a_mutated_package(tmp_path, fault, failing_test):
    result = run_on_mutated_package(tmp_path, TESTS, fault)
    assert_the_test_failed(result, TESTS, failing_test)
