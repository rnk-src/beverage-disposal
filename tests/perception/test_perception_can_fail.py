"""The camera-noise tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

NOISE_TESTS = "tests/perception/test_perception_noise.py"
EVAL_TESTS = "tests/policy/test_emptiness_eval.py"
SEQUENCER_TESTS = "tests/sequencer/test_sequencer.py"
PERCEPTION = "core/perception.py"
SEQUENCER = "core/sequencer.py"


def report_the_interval_without_the_bias(package):
    edit(
        package,
        PERCEPTION,
        "half = noise.interval_k * noise.dim_sigma_m + noise.dim_bias_max_m",
        "half = noise.interval_k * noise.dim_sigma_m",
    )


def draw_a_bias_for_every_length(package):
    edit(
        package,
        PERCEPTION,
        "estimate = true + bias + rng.normal(0.0, noise.dim_sigma_m)",
        "estimate = true + rng.uniform(-noise.dim_bias_max_m, noise.dim_bias_max_m)"
        " + rng.normal(0.0, noise.dim_sigma_m)",
    )


def let_a_length_go_negative(package):
    edit(
        package,
        PERCEPTION,
        "return (max(estimate - widen * half, 0.0), max(estimate + widen * half, 0.0))",
        "return (estimate - widen * half, estimate + widen * half)",
    )


def hide_the_level_of_every_labelled_bottle(package):
    edit(
        package,
        PERCEPTION,
        "hidden = shape.label_mass > 0 and low <= level <= high",
        "hidden = shape.label_mass > 0",
    )


def see_the_level_through_the_label(package):
    edit(package, PERCEPTION, "if dropped or hidden", "if dropped")


def never_drop_the_level(package):
    edit(package, PERCEPTION, "dropped = rng.random() < noise.level_dropout", "dropped = False")


def always_see_the_cap_right(package):
    edit(package, PERCEPTION, "if rng.random() < noise.cap_miss:", "if False:")


def always_see_the_base_right(package):
    edit(package, PERCEPTION, "if rng.random() < noise.base_style_error:", "if False:")


def trust_a_crushed_bottle_fully(package):
    edit(
        package,
        PERCEPTION,
        "confidence=1.0 - (noise.confidence_drop_crushed if deformed else 0.0),",
        "confidence=1.0,",
    )


def keep_a_crushed_bottles_width_interval(package):
    edit(package, PERCEPTION, "noise.deformed_width_factor if deformed else 1.0", "1.0")


def judge_on_the_truth_in_the_grid(package):
    edit(package, "core/emptiness_eval.py", "if noise is None\n", "if True\n")


def judge_on_the_truth_in_the_sequencer(package):
    edit(package, SEQUENCER, "if self.perception_noise is None:", "if True:")


def give_the_supervisor_the_true_width(package):
    edit(package, SEQUENCER, "if perception is None or item.bottle is None:", "if True:")


def look_the_same_way_for_every_seed(package):
    edit(
        package,
        SEQUENCER,
        "rng = np.random.default_rng([seed, PERCEPTION_STREAM])",
        "rng = np.random.default_rng(0)",
    )


FAULTS = [
    (
        report_the_interval_without_the_bias,
        NOISE_TESTS,
        "test_the_reported_interval_is_the_estimate_plus_and_minus_k_sigma_and_the_bias",
    ),
    (draw_a_bias_for_every_length, NOISE_TESTS, "test_one_look_shares_one_bias_across_its_lengths"),
    (let_a_length_go_negative, NOISE_TESTS, "test_no_reported_length_is_negative"),
    (
        hide_the_level_of_every_labelled_bottle,
        NOISE_TESTS,
        "test_a_level_outside_the_label_or_on_a_bottle_without_one_is_seen",
    ),
    (see_the_level_through_the_label, NOISE_TESTS, "test_a_level_behind_the_label_is_never_seen"),
    (
        never_drop_the_level,
        NOISE_TESTS,
        "test_the_level_of_an_unlabelled_bottle_is_seen_except_for_the_dropouts",
    ),
    (
        always_see_the_cap_right,
        NOISE_TESTS,
        "test_the_cap_and_the_base_are_reported_wrongly_at_their_rates",
    ),
    (
        always_see_the_base_right,
        NOISE_TESTS,
        "test_the_cap_and_the_base_are_reported_wrongly_at_their_rates",
    ),
    (
        trust_a_crushed_bottle_fully,
        NOISE_TESTS,
        "test_a_sound_bottle_is_seen_with_full_confidence_and_a_crushed_one_below_the_minimum",
    ),
    (
        keep_a_crushed_bottles_width_interval,
        NOISE_TESTS,
        "test_a_crushed_bottles_width_interval_is_three_times_as_wide",
    ),
    (
        judge_on_the_truth_in_the_grid,
        EVAL_TESTS,
        "test_a_noisy_camera_bins_fewer_empties_than_an_exact_one",
    ),
    (
        judge_on_the_truth_in_the_sequencer,
        SEQUENCER_TESTS,
        "test_with_camera_noise_the_bottle_is_judged_on_what_the_camera_reports",
    ),
    (
        give_the_supervisor_the_true_width,
        SEQUENCER_TESTS,
        "test_the_supervisor_gets_the_grasp_width_the_camera_reports_as_an_interval",
    ),
    (look_the_same_way_for_every_seed, SEQUENCER_TESTS, "test_the_cameras_look_follows_the_seed"),
]


@pytest.mark.medium
@pytest.mark.parametrize(
    ("fault", "tests", "failing_test"), FAULTS, ids=lambda value: getattr(value, "__name__", None)
)
def test_the_camera_tests_fail_on_a_mutated_package(tmp_path, fault, tests, failing_test):
    result = run_on_mutated_package(tmp_path, tests, fault)
    assert_the_test_failed(result, tests, failing_test)
