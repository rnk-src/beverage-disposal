"""The fullness tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

STATS_TESTS = "tests/fullness/test_statistics.py"
PAD_TESTS = "tests/fullness/test_pad_sensor.py"
RULE_TESTS = "tests/fullness/test_fullness_rule.py"
SWEEP_TESTS = "tests/fullness/test_false_bin_sweep.py"
STATS = "core/statistics.py"
PAD = "core/pad.py"
RULE = "core/fullness.py"
CONFIG = "core/fullness_config.py"
EVAL = "core/fullness_eval.py"


def drop_the_continuity_term(package):
    edit(package, STATS, "denominator = 1 + z * z / trials", "denominator = 1")


def settle_in_one_second_whatever_the_time_constant(package):
    edit(package, PAD, "math.exp(-t / c.settle_time_constant_s)", "math.exp(-t)")


def ignore_the_drift(package):
    edit(package, PAD, "+ c.drift_g_per_s * t", "+ 0.0")


def keep_the_arm_load_forever(package):
    edit(package, PAD, "c.arm_load_g * math.exp(-t / c.arm_load_time_constant_s)", "c.arm_load_g")


def draw_a_new_offset_for_every_sample(package):
    edit(
        package,
        PAD,
        "self._rng.normal(0.0, c.offset_sigma_g, size=(n, 1))",
        "self._rng.normal(0.0, c.offset_sigma_g, size=(n, len(times)))",
    )


def skip_the_quantization(package):
    edit(package, PAD, "        if c.resolution_g > 0:", "        if False:")


def call_every_weighing_stable(package):
    edit(package, PAD, "moving <= c.stability_limit_g", "moving >= 0")


def accept_a_negative_mass(package):
    edit(package, PAD, "        if true_mass_kg < 0:", "        if False:")


def raise_the_limit_by_a_gram(package):
    edit(package, RULE, "(net_g < rule.limit_g)", "(net_g < rule.limit_g + 1.0)")


def forget_the_margin_in_the_limit(package):
    edit(package, CONFIG, "- self.margin_g", "- 0.0")


def bin_an_unknown_container(package):
    edit(package, RULE, "    if record is None:", "    if False:")


def ignore_the_cap(package):
    edit(
        package,
        RULE,
        "    if record.cap_check_required and cap_confirmed is not True:",
        "    if False:",
    )


def trust_a_reading_below_the_lightest_empty_weight(package):
    edit(package, RULE, "    if net_g < -rule.margin_g:", "    if False:")


def ignore_an_unsettled_weight(package):
    edit(package, RULE, "    if not weighing.stable:", "    if False:")


def forget_the_tare_in_the_simulation(package):
    edit(
        package,
        EVAL,
        "true_mass = empty_mass_kg + volume_ml * density / 1000.0",
        "true_mass = volume_ml * density / 1000.0",
    )


FAULTS = [
    (
        STATS_TESTS,
        drop_the_continuity_term,
        "test_no_failures_in_381_trials_bounds_the_rate_just_under_one_percent",
    ),
    (
        PAD_TESTS,
        settle_in_one_second_whatever_the_time_constant,
        "test_the_signal_rises_like_a_first_order_lag",
    ),
    (PAD_TESTS, ignore_the_drift, "test_drift_adds_the_rate_times_the_elapsed_time"),
    (PAD_TESTS, keep_the_arm_load_forever, "test_the_arm_load_decays_with_its_time_constant"),
    (
        PAD_TESTS,
        draw_a_new_offset_for_every_sample,
        "test_the_zero_offset_is_shared_by_one_weighings_samples_but_varies_between_weighings",
    ),
    (
        PAD_TESTS,
        skip_the_quantization,
        "test_readings_are_multiples_of_the_resolution_before_averaging",
    ),
    (PAD_TESTS, call_every_weighing_stable, "test_a_pad_still_moving_is_reported_unstable"),
    (
        PAD_TESTS,
        accept_a_negative_mass,
        "test_an_empty_pad_reads_about_zero_and_a_negative_mass_is_rejected",
    ),
    (
        RULE_TESTS,
        raise_the_limit_by_a_gram,
        "test_a_container_just_under_the_limit_is_binned_and_just_over_is_set_aside",
    ),
    (
        RULE_TESTS,
        forget_the_margin_in_the_limit,
        "test_the_rule_file_says_thirty_millilitres_at_the_lightest_drink_less_a_margin",
    ),
    (RULE_TESTS, bin_an_unknown_container, "test_a_container_of_an_unknown_type_is_set_aside"),
    (RULE_TESTS, ignore_the_cap, "test_a_bottle_needs_its_cap_confirmed_before_it_can_be_binned"),
    (
        RULE_TESTS,
        trust_a_reading_below_the_lightest_empty_weight,
        "test_a_reading_well_below_the_lightest_empty_weight_is_not_trusted",
    ),
    (RULE_TESTS, ignore_an_unsettled_weight, "test_a_weighing_that_did_not_settle_is_set_aside"),
    (
        SWEEP_TESTS,
        forget_the_tare_in_the_simulation,
        "test_an_empty_container_is_binned_almost_every_time_even_at_a_heavier_tare[can]",
    ),
]


@pytest.mark.parametrize(
    ("module", "fault", "failing_test"),
    FAULTS,
    ids=lambda value: getattr(value, "__name__", None) if callable(value) else None,
)
def test_the_fullness_tests_fail_on_a_mutated_package(tmp_path, module, fault, failing_test):
    result = run_on_mutated_package(tmp_path, module, fault)
    assert_the_test_failed(result, module, failing_test)
