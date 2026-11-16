"""The classifier tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

TESTS = "tests/train/test_classifier.py"
CLASSIFIER = "core/classifier.py"
BUNDLE = "train/emptiness/bundle.py"


def leave_the_range_ends_outside(package):
    edit(
        package,
        CLASSIFIER,
        "inside = (x >= np.array(self.lo)) & (x <= np.array(self.hi))",
        "inside = (x > np.array(self.lo)) & (x < np.array(self.hi))",
    )


def ignore_the_upper_end(package):
    edit(package, CLASSIFIER, "& (x <= np.array(self.hi))", "")


def let_any_feature_be_missing(package):
    edit(
        package,
        CLASSIFIER,
        "np.where(missing, np.array(self.may_be_missing), inside)",
        "np.where(missing, True, inside)",
    )


def accept_a_vector_of_any_length(package):
    edit(package, CLASSIFIER, "if x.shape != (len(self.lo),):", "if False:")


def skip_the_checksum(package):
    edit(
        package,
        BUNDLE,
        'if hashlib.sha256(blob).hexdigest() != record["model_sha256"]:',
        "if False:",
    )


def skip_the_version_check(package):
    edit(package, BUNDLE, 'if record["meta"]["sklearn"] != sklearn.__version__:', "if False:")


def skip_the_feature_name_check(package):
    edit(package, BUNDLE, 'if tuple(record["feature_names"]) != tuple(feature_names):', "if False:")


def score_the_chance_of_being_over(package):
    edit(package, "train/emptiness/sklearn_model.py", "EMPTY_CLASS = 1", "EMPTY_CLASS = 0")


def fit_on_the_surely_over_rows_too(package):
    edit(
        package,
        "train/emptiness/fit.py",
        "rows = np.isin(train.zone, FIT_ZONES)",
        "rows = np.ones(len(train.zone), bool)",
    )


FAULTS = [
    (
        leave_the_range_ends_outside,
        "test_a_vector_inside_every_range_is_in_distribution_and_the_ends_count_as_inside",
    ),
    (ignore_the_upper_end, "test_a_vector_with_one_feature_outside_its_range_is_not[vector1]"),
    (let_any_feature_be_missing, "test_a_missing_value_is_fine_only_where_missing_is_ordinary"),
    (accept_a_vector_of_any_length, "test_a_vector_of_the_wrong_length_is_not_in_distribution"),
    (skip_the_checksum, "test_a_model_file_with_one_byte_changed_is_refused"),
    (skip_the_version_check, "test_a_bundle_from_another_library_version_is_refused"),
    (skip_the_feature_name_check, "test_a_bundle_made_for_other_features_is_refused"),
    (
        score_the_chance_of_being_over,
        "test_in_the_ambiguous_zone_the_model_scores_emptier_bottles_higher",
    ),
    (fit_on_the_surely_over_rows_too, "test_the_fit_never_sees_a_surely_over_row"),
]


@pytest.mark.medium
@pytest.mark.parametrize(
    ("fault", "failing_test"), FAULTS, ids=lambda value: getattr(value, "__name__", None)
)
def test_the_classifier_tests_fail_on_a_mutated_package(tmp_path, fault, failing_test):
    result = run_on_mutated_package(tmp_path, TESTS, fault, only=failing_test)
    assert_the_test_failed(result, TESTS, failing_test)
