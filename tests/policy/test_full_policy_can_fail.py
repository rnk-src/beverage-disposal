"""The full-policy tests are run against mutated copies of the policy and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

TESTS = "tests/policy/test_full_policy.py"
POLICY = "core/policy.py"
OVERTURN = "if self.config.allow_overturn and score > tau:"
TAU = "tau = max(self.bundle.tau_by_size.get(name, math.inf) for name in found.matches)"


def change(old, new):
    def fault(package):
        edit(package, POLICY, old, new)

    return fault


FAULTS = {
    "veto_a_score_at_the_veto_score": (
        change("if score < self.config.veto_score:", "if score <= self.config.veto_score:"),
        "test_a_score_exactly_at_the_veto_score_does_not_veto",
    ),
    "fix_the_veto_score": (
        change("if score < self.config.veto_score:", "if score < 0.5:"),
        "test_the_veto_score_comes_from_the_config",
    ),
    "never_let_the_classifier_veto": (
        change("if score < self.config.veto_score:", "if False:"),
        "test_a_score_under_the_veto_score_sets_a_surely_under_bottle_aside",
    ),
    "bin_a_score_at_the_threshold": (
        change(OVERTURN, OVERTURN.replace("score > tau", "score >= tau")),
        "test_a_score_exactly_at_the_threshold_does_not_bin",
    ),
    "overturn_on_hardware_too": (
        change(OVERTURN, OVERTURN.replace("self.config.allow_overturn and ", "")),
        "test_without_allow_overturn_the_ambiguous_zone_is_never_binned",
    ),
    "take_the_looser_of_two_thresholds": (
        change(TAU, TAU.replace("tau = max(", "tau = min(")),
        "test_a_bottle_that_fits_two_sizes_must_beat_the_stricter_of_their_thresholds[pet_330ml]",
    ),
    "take_the_first_matching_sizes_threshold": (
        change(TAU, TAU.replace("in found.matches)", "in found.matches[:1])")),
        "test_a_bottle_that_fits_two_sizes_must_beat_the_stricter_of_their_thresholds[pet_330ml]",
    ),
    "take_the_last_matching_sizes_threshold": (
        change(TAU, TAU.replace("in found.matches)", "in found.matches[-1:])")),
        "test_a_bottle_that_fits_two_sizes_must_beat_the_stricter_of_their_thresholds[pet_250ml]",
    ),
    "give_a_size_without_a_threshold_zero": (
        change(TAU, TAU.replace("math.inf", "0.0")),
        "test_a_size_without_a_threshold_is_never_overturned",
    ),
    "skip_a_matching_size_without_a_threshold": (
        change(TAU, TAU.replace("math.inf", "-math.inf")),
        "test_a_bottle_that_fits_a_size_without_a_threshold_is_not_overturned[pet_330ml]",
    ),
    "use_the_first_sizes_threshold": (
        change(
            TAU,
            TAU.replace("for name in found.matches", "for name in list(self.family.classes)[:1]"),
        ),
        "test_the_threshold_is_that_of_the_bottles_own_size",
    ),
    "trust_the_model_outside_its_ranges": (
        change("if not self.bundle.ranges.in_distribution(x):", "if False:"),
        "test_a_bottle_outside_the_training_ranges_is_vetoed_and_the_model_not_asked[ambiguous]",
    ),
    "take_nan_for_a_score": (
        change("if math.isnan(score):", "if False:"),
        "test_a_model_that_raises_or_scores_nan_sets_the_bottle_aside[surely_under-nan]",
    ),
    "let_a_model_error_through": (
        change("except Exception as error:", "except KeyError as error:"),
        "test_a_model_that_raises_or_scores_nan_sets_the_bottle_aside[ambiguous-raises]",
    ),
    "ask_the_model_about_a_surely_over_bottle": (
        change(
            "if zone is Zone.SURELY_OVER:", "if zone is Zone.SURELY_OVER and self.bundle is None:"
        ),
        "test_the_model_is_not_even_asked_about_a_surely_over_bottle",
    ),
    "let_the_classifier_bin_a_surely_over_bottle": (
        change(
            "if zone is Zone.SURELY_OVER:", "if zone is Zone.SURELY_OVER and self.bundle is None:"
        ),
        "test_no_classifier_changes_a_veto_or_the_surely_over_zone",
    ),
    "drop_the_score_from_the_verdict": (
        change(
            'return verdict(SET_ASIDE, "the weight cannot tell", score=score)',
            'return verdict(SET_ASIDE, "the weight cannot tell")',
        ),
        "test_a_score_exactly_at_the_threshold_does_not_bin",
    ),
}


@pytest.mark.medium
@pytest.mark.parametrize("name", FAULTS)
def test_the_full_policy_tests_fail_on_a_mutated_policy(tmp_path, name):
    fault, failing_test = FAULTS[name]
    result = run_on_mutated_package(tmp_path, TESTS, fault)
    assert_the_test_failed(result, TESTS, failing_test)


EVAL_TESTS = "tests/policy/test_emptiness_eval.py"
EVAL = "core/emptiness_eval.py"


def change_eval(old, new):
    def fault(package):
        edit(package, EVAL, old, new)

    return fault


GRID_FAULTS = {
    "ignore_the_grids_stream": (
        change_eval("default_rng([seed, *stream, index,", "default_rng([seed, index,"),
        "test_a_stream_gives_the_grid_bottles_of_its_own",
    ),
    "draw_sound_bottles_for_a_crushed_cell": (
        change_eval("deformed=deformed, foreign_g=foreign_g", "foreign_g=foreign_g"),
        "test_every_crushed_bottle_of_the_grid_is_vetoed_and_the_cell_says_it_was_crushed",
    ),
    "leave_the_foreign_object_out_of_the_grid": (
        change_eval("deformed=deformed, foreign_g=foreign_g", "deformed=deformed"),
        "test_a_foreign_object_of_the_grid_is_on_the_scale_and_named_in_the_cell",
    ),
    "count_a_classifiers_veto_as_an_overturn": (
        change_eval(
            "overturned += verdict.action == BIN and verdict.decided_by == CLASSIFIER",
            "overturned += verdict.decided_by == CLASSIFIER",
        ),
        "test_a_classifiers_veto_is_not_counted_as_an_overturn",
    ),
    "count_every_bin_as_an_overturn": (
        change_eval(
            "overturned += verdict.action == BIN and verdict.decided_by == CLASSIFIER",
            "overturned += verdict.action == BIN",
        ),
        "test_a_cell_counts_the_bins_the_classifier_made_in_the_ambiguous_zone",
    ),
    "test_the_nominal_size_over_the_payload": (
        change_eval("return min(family.classes[size].nominal_ml,", "return max(0.0,"),
        "test_the_payload_volume_is_the_nominal_size_or_what_the_arm_may_lift",
    ),
    "call_exactly_the_threshold_under": (
        change_eval("if c.volume_ml >= threshold_ml]", "if c.volume_ml > threshold_ml]"),
        "test_the_summary_counts_misbins_empties_overturns_and_vetoes_by_hand",
    ),
    "count_crushed_bottles_among_the_sound": (
        change_eval("if not c.deformed and c.foreign_g == 0.0]", "if c.foreign_g == 0.0]"),
        "test_a_crushed_or_foreign_cell_is_not_counted_among_the_sound_bottles",
    ),
    "count_foreign_objects_among_the_empties": (
        change_eval("if not c.deformed and c.foreign_g == 0.0]", "if not c.deformed]"),
        "test_a_crushed_or_foreign_cell_is_not_counted_among_the_sound_bottles",
    ),
    "report_the_best_bound_as_the_worst": (
        change_eval("max((c.upper_95 for c in over)", "min((c.upper_95 for c in over)"),
        "test_the_summary_counts_misbins_empties_overturns_and_vetoes_by_hand",
    ),
}


@pytest.mark.medium
@pytest.mark.parametrize("name", GRID_FAULTS)
def test_the_grid_tests_fail_on_a_mutated_grid(tmp_path, name):
    fault, failing_test = GRID_FAULTS[name]
    result = run_on_mutated_package(tmp_path, EVAL_TESTS, fault)
    assert_the_test_failed(result, EVAL_TESTS, failing_test)


DATASET_TESTS = "tests/train/test_dataset.py"


def spread_the_threshold_cell_over_its_bin(package):
    edit(
        package,
        "train/emptiness/dataset.py",
        "at_threshold = low == policy.rule.threshold_ml",
        "at_threshold = False",
    )


@pytest.mark.medium
def test_the_dataset_tests_notice_a_threshold_cell_that_is_not_at_the_threshold(tmp_path):
    result = run_on_mutated_package(tmp_path, DATASET_TESTS, spread_the_threshold_cell_over_its_bin)
    assert_the_test_failed(
        result,
        DATASET_TESTS,
        "test_the_cell_at_the_threshold_holds_bottles_of_exactly_the_threshold",
    )
