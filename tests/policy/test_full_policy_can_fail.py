"""The full-policy tests are run against mutated copies of the policy and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

TESTS = "tests/policy/test_full_policy.py"
POLICY = "core/policy.py"
OVERTURN = "if self.config.allow_overturn and len(found.matches) == 1 and score > tau:"


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
    "overturn_a_bottle_of_two_sizes": (
        change(OVERTURN, OVERTURN.replace("len(found.matches) == 1 and ", "")),
        "test_a_bottle_that_fits_two_sizes_is_not_overturned",
    ),
    "give_a_size_without_a_threshold_zero": (
        change("get(found.matches[0], math.inf)", "get(found.matches[0], 0.0)"),
        "test_a_size_without_a_threshold_is_never_overturned",
    ),
    "use_the_first_sizes_threshold": (
        change("get(found.matches[0], math.inf)", "get(next(iter(self.family.classes)), math.inf)"),
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
