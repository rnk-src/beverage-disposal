"""The calibration tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

TESTS = "tests/npcal/test_np_calibration.py"
PIPELINE_TESTS = "tests/train/test_pipeline.py"
NPCAL = "core/np_calibration.py"
PIPELINE = "train/emptiness/pipeline.py"
SAFE_ORDER = "test_with_more_samples_the_threshold_is_the_smallest_order_statistic_that_is_safe"


def take_the_order_statistic_below(package):
    edit(package, NPCAL, "return float(ordered[low - 1])", "return float(ordered[low - 2])")


def take_the_plain_quantile(package):
    edit(
        package,
        NPCAL,
        "if violation_probability(n, middle, alpha) <= delta:",
        "if middle >= math.ceil((1 - alpha) * n):",
    )


def accept_too_few_samples(package):
    edit(package, NPCAL, "if n < needed:", "if False:")


def round_the_sample_size_down(package):
    edit(package, NPCAL, "return math.ceil(math.log(delta)", "return math.floor(math.log(delta)")


def sum_the_tail_from_one_term_late(package):
    edit(package, NPCAL, "j = np.arange(k, n + 1)", "j = np.arange(k + 1, n + 1)")


def take_the_mildest_cell(package):
    edit(
        package,
        NPCAL,
        "taus[size] = max(tau, taus.get(size, -math.inf))",
        "taus[size] = min(tau, taus.get(size, math.inf))",
    )


def put_an_edge_volume_in_the_cell_below(package):
    edit(
        package, NPCAL, "(volumes >= low) & (volumes < high)", "(volumes > low) & (volumes <= high)"
    )


def accept_rows_under_the_threshold(package):
    edit(package, NPCAL, "if (volumes < bins[0]).any():", "if False:")


def calibrate_on_the_training_split(package):
    edit(
        package,
        PIPELINE,
        'generate_cells(data, policy, pad, noise, split="calibration", seed=seed)',
        'generate_cells(data, policy, pad, noise, split="train", seed=seed)',
    )


def give_a_short_size_a_threshold_of_zero(package):
    edit(
        package,
        PIPELINE,
        "taus.update({size: math.inf for size in short})",
        "taus.update({size: 0.0 for size in short})",
    )


FAULTS = [
    (take_the_order_statistic_below, TESTS, SAFE_ORDER),
    (take_the_plain_quantile, TESTS, SAFE_ORDER),
    (accept_too_few_samples, TESTS, "test_too_few_samples_are_refused"),
    (
        round_the_sample_size_down,
        TESTS,
        "test_the_fewest_samples_are_the_first_n_with_all_n_below_alpha_unlikely_enough",
    ),
    (
        sum_the_tail_from_one_term_late,
        TESTS,
        "test_the_violation_probability_is_a_binomial_tail[10]",
    ),
    (take_the_mildest_cell, TESTS, "test_a_sizes_threshold_is_the_strictest_of_its_cells"),
    (
        put_an_edge_volume_in_the_cell_below,
        TESTS,
        "test_empty_cells_are_listed_and_a_volume_on_an_edge_belongs_to_the_cell_above",
    ),
    (
        accept_rows_under_the_threshold,
        TESTS,
        "test_rows_under_the_first_edge_are_not_calibration_rows",
    ),
    (
        calibrate_on_the_training_split,
        PIPELINE_TESTS,
        "test_the_thresholds_come_from_the_calibration_split_and_not_the_training_rows",
    ),
    (
        give_a_short_size_a_threshold_of_zero,
        PIPELINE_TESTS,
        "test_a_size_with_a_cell_too_small_to_calibrate_can_never_be_overturned",
    ),
]


@pytest.mark.medium
@pytest.mark.parametrize(
    ("fault", "tests", "failing_test"), FAULTS, ids=lambda value: getattr(value, "__name__", None)
)
def test_the_calibration_tests_fail_on_a_mutated_package(tmp_path, fault, tests, failing_test):
    result = run_on_mutated_package(tmp_path, tests, fault)
    assert_the_test_failed(result, tests, failing_test)
