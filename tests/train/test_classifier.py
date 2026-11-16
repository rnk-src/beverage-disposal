"""The classifier: the range check, a deterministic fit, the saved bundle, and a core without it."""

import functools
import json
import math
import subprocess
import sys

import numpy as np
import pytest
import sklearn
from test_dataset import FULLNESS, NAMES, NOISE, POLICY

from beverage_disposal.core.classifier import FeatureRanges
from beverage_disposal.train.emptiness.bundle import (
    BUNDLE_FILE,
    MODEL_FILE,
    load_bundle,
    save_bundle,
)
from beverage_disposal.train.emptiness.dataset import DatasetConfig, generate
from beverage_disposal.train.emptiness.fit import FitConfig, fit

RANGES = FeatureRanges(lo=(0.0, 1.0, 5.0), hi=(1.0, 2.0, 9.0), may_be_missing=(False, False, True))
CONFIG = DatasetConfig(
    sizes=("pet_250ml", "pet_500ml"), rows_per_size=2500, volume_bins=(30.0, 31.0, 35.0, 50.0)
)
QUICK = FitConfig(max_iter=40)


@functools.cache
def rows(split):
    return generate(CONFIG, POLICY, FULLNESS.pad, NOISE, split=split, seed=0)


@functools.cache
def model():
    return fit(rows("train"), QUICK)


def test_a_vector_inside_every_range_is_in_distribution_and_the_ends_count_as_inside():
    assert RANGES.in_distribution(np.array([0.5, 1.5, 7.0]))
    assert RANGES.in_distribution(np.array([0.0, 2.0, 9.0]))


@pytest.mark.parametrize("vector", [[-0.01, 1.5, 7.0], [0.5, 2.01, 7.0], [0.5, 1.5, 9.5]])
def test_a_vector_with_one_feature_outside_its_range_is_not(vector):
    assert not RANGES.in_distribution(np.array(vector))


def test_a_missing_value_is_fine_only_where_missing_is_ordinary():
    assert RANGES.in_distribution(np.array([0.5, 1.5, math.nan]))
    assert not RANGES.in_distribution(np.array([math.nan, 1.5, 7.0]))
    assert not RANGES.in_distribution(np.array([0.5, math.inf, 7.0]))


def test_a_vector_of_the_wrong_length_is_not_in_distribution():
    assert not RANGES.in_distribution(np.array([0.5, 1.5]))


def test_the_ranges_of_some_rows_are_their_smallest_and_largest_values_missing_ones_aside():
    table = np.array([[1.0, 5.0, math.nan], [3.0, 4.0, 2.0], [2.0, 6.0, 8.0]])
    ranges = FeatureRanges.from_rows(table, ("w_g", "v_hi", "level_v_hi"))
    assert (ranges.lo, ranges.hi) == ((1.0, 4.0, 2.0), (3.0, 6.0, 8.0))
    assert ranges.may_be_missing == (False, False, True)


def test_rows_with_a_value_missing_where_it_must_not_be_are_refused():
    with pytest.raises(ValueError, match="w_g"):
        FeatureRanges.from_rows(np.array([[math.nan, 1.0]]), ("w_g", "v_hi"))


@pytest.mark.medium
def test_two_fits_of_the_same_rows_score_identically():
    again = fit(rows("train"), QUICK)
    test = rows("test").X
    assert model().scores(test).tobytes() == again.scores(test).tobytes()


@pytest.mark.medium
def test_the_score_of_one_row_is_its_entry_in_the_scores_of_all():
    test = rows("test").X
    scores = model().scores(test)
    assert model().score(test[3]) == scores[3]
    assert ((scores >= 0) & (scores <= 1)).all()


@pytest.mark.medium
def test_in_the_ambiguous_zone_the_model_scores_emptier_bottles_higher():
    """Catches a model that learned nothing the weight rule does not know already."""
    test = rows("test")
    ambiguous = test.zone == "ambiguous"
    scores = model().scores(test.X)
    under, over = scores[ambiguous & test.y], scores[ambiguous & ~test.y]
    assert len(under) > 100 and len(over) > 100
    assert under.mean() > over.mean() + 0.2
    # Bottles holding under 10 mL against bottles of 50 mL or more: hardly ever the wrong way.
    low = scores[ambiguous & (test.volume < 10)]
    high = scores[ambiguous & (test.volume >= 50)]
    assert np.median(low) > np.quantile(high, 0.95)


@pytest.mark.medium
def test_the_fit_never_sees_a_surely_over_row():
    """A fit on the rows with surely-over ones removed by hand gives the same model."""
    train = rows("train")
    keep = train.zone != "surely_over"
    assert not keep.all()
    by_hand = type(train)(
        train.X[keep],
        train.y[keep],
        train.volume[keep],
        train.size[keep],
        train.zone[keep],
        train.names,
        train.seed,
    )
    test = rows("test").X
    assert fit(by_hand, QUICK).scores(test).tobytes() == model().scores(test).tobytes()


def saved(tmp_path):
    ranges = FeatureRanges.from_rows(rows("train").X, NAMES)
    return save_bundle(model(), {"pet_250ml": 0.9}, ranges, NAMES, {"seed": 0}, tmp_path)


@pytest.mark.medium
def test_a_saved_bundle_loads_back_with_the_same_scores_thresholds_and_ranges(tmp_path):
    written = saved(tmp_path)
    loaded = load_bundle(tmp_path, NAMES)
    test = rows("test").X
    assert [loaded.model.score(row) for row in test[:50]] == [
        model().score(row) for row in test[:50]
    ]
    assert loaded.tau_by_size == {"pet_250ml": 0.9}
    assert loaded.ranges == written.ranges and loaded.feature_names == NAMES
    assert loaded.model_sha256 == written.model_sha256 and len(loaded.model_sha256) == 64
    assert loaded.meta["seed"] == 0 and loaded.meta["sklearn"] == sklearn.__version__


@pytest.mark.medium
def test_a_model_file_with_one_byte_changed_is_refused(tmp_path):
    saved(tmp_path)
    path = tmp_path / MODEL_FILE
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path, NAMES)


@pytest.mark.medium
def test_a_bundle_made_for_other_features_is_refused(tmp_path):
    saved(tmp_path)
    with pytest.raises(ValueError, match="feature"):
        load_bundle(tmp_path, (*NAMES[1:], NAMES[0]))


@pytest.mark.medium
def test_a_bundle_from_another_library_version_is_refused(tmp_path):
    saved(tmp_path)
    path = tmp_path / BUNDLE_FILE
    record = json.loads(path.read_text())
    record["meta"]["sklearn"] = "0.0.1"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="scikit-learn"):
        load_bundle(tmp_path, NAMES)


WITHOUT_SKLEARN = """
import sys
sys.modules["sklearn"] = None
sys.modules["mujoco"] = None
import beverage_disposal.core.classifier, beverage_disposal.core.policy
import beverage_disposal.core.features, beverage_disposal.core.sequencer
print("ok")
"""


def test_the_core_imports_with_the_learning_library_unavailable():
    result = subprocess.run([sys.executable, "-c", WITHOUT_SKLEARN], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
