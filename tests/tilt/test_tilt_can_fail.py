"""The tilt tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

COINFT_TESTS = "tests/tilt/test_coinft.py"
TILT_TESTS = "tests/tilt/test_tilt_routine.py"
COINFT = "core/coinft.py"
TILT = "core/tilt.py"


def draw_an_offset_for_every_sample(package):
    edit(
        package,
        COINFT,
        "self._rng.normal(0.0, c.offset_sigma_mnm, size=(n, 1))",
        "self._rng.normal(0.0, c.offset_sigma_mnm, size=(n, c.samples_averaged))",
    )


def skip_the_least_count(package):
    edit(package, COINFT, "        if c.resolution_mnm > 0:", "        if False:")


def flip_the_empty_containers_lever(package):
    edit(package, TILT, "    lever = -(empty_centre_fraction", "    lever = (empty_centre_fraction")


def forget_the_empty_centre_error(package):
    edit(package, TILT, "+ shape + empty_centre", "+ shape")


def forget_the_sensor_error(package):
    edit(package, TILT, "test.k_sigma * sigma_mnm / 1000.0 +", "0.0 +")


def bin_anything_the_bound_allows(package):
    edit(
        package,
        TILT,
        "if bound < config.test.threshold_ml:",
        "if bound < config.test.threshold_ml + 100.0:",
    )


def skip_the_weight_check(package):
    edit(
        package, TILT, "    if gross_mass_kg > config.test.unknown_gross_limit_kg:", "    if False:"
    )


def trust_a_reading_outside_the_model(package):
    edit(package, TILT, "    if bound is None:", "    if False:")


def let_the_pad_alone_bin_heavy_containers(package):
    edit(package, TILT, "    if gross_g < rule.limit_g:", "    if gross_g < rule.limit_g + 100.0:")


FAULTS = [
    (COINFT_TESTS, draw_an_offset_for_every_sample, "test_the_zero_offset_is_not_averaged_away"),
    (COINFT_TESTS, skip_the_least_count, "test_one_sample_is_rounded_to_the_least_count"),
    (
        TILT_TESTS,
        flip_the_empty_containers_lever,
        "test_an_empty_container_loads_the_grasp_only_through_its_own_weight",
    ),
    (
        TILT_TESTS,
        forget_the_empty_centre_error,
        "test_the_uncertainty_adds_the_sensor_the_shape_errors_and_the_empty_centre",
    ),
    (TILT_TESTS, forget_the_sensor_error, "test_a_looser_sensor_makes_the_bound_wider"),
    (
        TILT_TESTS,
        bin_anything_the_bound_allows,
        "test_a_can_at_or_over_thirty_millilitres_read_without_noise_is_set_aside[31.0]",
    ),
    (
        TILT_TESTS,
        skip_the_weight_check,
        "test_a_container_heavier_than_the_unknown_limit_is_set_aside_without_looking_at_the_torque",
    ),
    (
        TILT_TESTS,
        trust_a_reading_outside_the_model,
        "test_a_reading_far_outside_what_the_model_allows_is_set_aside",
    ),
    (
        TILT_TESTS,
        let_the_pad_alone_bin_heavy_containers,
        "test_the_pad_alone_bins_an_unknown_container_only_when_its_whole_weight_is_under_the_limit",
    ),
]


@pytest.mark.parametrize(
    ("module", "fault", "failing_test"),
    FAULTS,
    ids=lambda value: getattr(value, "__name__", None) if callable(value) else None,
)
def test_the_tilt_tests_fail_on_a_mutated_package(tmp_path, module, fault, failing_test):
    result = run_on_mutated_package(tmp_path, module, fault)
    assert_the_test_failed(result, module, failing_test)
