"""The classifier's simulated dataset: labels, seeds, disjoint splits and the calibration cells."""

import numpy as np
import pytest
from support.paths import ROOT

from beverage_disposal.core.bottle_family import load_family
from beverage_disposal.core.features import feature_names
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.perception import load_perception_noise
from beverage_disposal.core.policy import EmptinessPolicy, load_policy_config
from beverage_disposal.train.emptiness.dataset import (
    DatasetConfig,
    cell_edges,
    generate,
    generate_cells,
)

CONFIGS = ROOT / "configs"
FULLNESS = load_fullness_config(CONFIGS / "fullness.toml")
FAMILY = load_family(CONFIGS / "bottles.toml")
PET = load_materials(CONFIGS / "materials.toml")["PET"]
NOISE = load_perception_noise(CONFIGS / "perception.toml")
POLICY = EmptinessPolicy(
    FULLNESS.rule, PET, FAMILY, load_policy_config(CONFIGS / "policy.toml"), tuple(FAMILY.classes)
)
SMALL = DatasetConfig(
    sizes=("pet_250ml", "pet_2000ml"),
    rows_per_size=150,
    volume_bins=(30.0, 31.0, 35.0, 50.0),
    rows_per_cell=20,
    max_draws_per_cell=4000,
)
NAMES = feature_names(FAMILY)


def mixture(split="train", seed=0, config=SMALL):
    return generate(config, POLICY, FULLNESS.pad, NOISE, split=split, seed=seed)


def cells(split="calibration", seed=0, config=SMALL):
    return generate_cells(config, POLICY, FULLNESS.pad, NOISE, split=split, seed=seed)


def column(data, name):
    return data.X[:, NAMES.index(name)]


def test_the_cells_of_a_size_end_at_what_it_can_hold_or_the_arm_can_carry():
    assert cell_edges(SMALL, 250.0) == [(30.0, 31.0), (31.0, 35.0), (35.0, 50.0), (50.0, 250.0)]
    # A 2 L bottle is over the 0.6 kg payload long before it is full.
    assert cell_edges(SMALL, 2000.0)[-1] == (50.0, 600.0)


def test_a_mixture_has_one_row_per_kept_bottle_with_the_feature_names():
    data = mixture()
    n = len(data.y)
    assert 200 < n <= 300  # 150 per size, less the vetoed ones
    assert data.X.shape == (n, len(NAMES)) and data.names == NAMES
    assert len(data.volume) == len(data.size) == len(data.zone) == n
    assert set(data.size) == {"pet_250ml", "pet_2000ml"}
    assert set(data.zone) <= {"surely_under", "ambiguous", "surely_over"}
    assert data.seed == (0, 0)


def test_the_label_is_whether_the_true_volume_is_under_thirty():
    data = mixture()
    assert data.y.dtype == bool
    assert np.array_equal(data.y, data.volume < 30.0)
    assert data.y.any() and not data.y.all()


def test_the_mixture_covers_the_threshold_and_the_cells_beyond_it():
    volume = mixture().volume
    assert (volume < 60.0).mean() > 0.5
    assert (volume > 60.0).sum() > 10 and volume.max() <= 600.0
    assert volume.min() >= 0.0


def test_the_rows_are_what_the_policy_works_out_not_the_truth():
    """Catches features made from the true bottle: the most liquid follows the pad weight."""
    data = mixture()
    rule = FULLNESS.rule
    most = (
        column(data, "w_g") - column(data, "emin_g") + rule.margin_g
    ) / rule.min_density_g_per_ml
    assert column(data, "v_hi") == pytest.approx(most)
    # The camera's intervals are 18 mm wide (4 sigma of 2 mm plus 1 mm of bias, each way).
    assert column(data, "h_hi") - column(data, "h_lo") == pytest.approx(0.018)


def test_no_surely_under_row_holds_thirty_or_more_and_no_surely_over_row_less():
    data = mixture()
    assert (data.volume[data.zone == "surely_under"] < 30.0).all()
    assert (data.volume[data.zone == "surely_over"] >= 30.0).all()


def test_the_same_seed_and_split_give_the_same_bytes():
    first, second = mixture(seed=5), mixture(seed=5)
    assert first.X.tobytes() == second.X.tobytes()
    assert first.volume.tobytes() == second.volume.tobytes()
    assert mixture(seed=6).X.tobytes() != first.X.tobytes()


def test_the_splits_share_no_bottle():
    rows = {
        split: {row.tobytes() for row in mixture(split=split).X}
        for split in ("train", "calibration", "test", "acceptance")
    }
    names = list(rows)
    for i, one in enumerate(names):
        for other in names[i + 1 :]:
            assert not rows[one] & rows[other], (one, other)


def test_an_unknown_split_is_refused():
    with pytest.raises(ValueError, match="split"):
        mixture(split="validation")


def test_every_filled_calibration_cell_has_exactly_the_rows_asked_all_ambiguous_and_in_its_bin():
    data, counts = cells()
    assert set(data.zone) == {"ambiguous"}
    assert set(counts) == {(size, low) for size in SMALL.sizes for low in (30.0, 31.0, 35.0, 50.0)}
    for (size, low), count in counts.items():
        nominal = FAMILY.classes[size].nominal_ml
        high = dict(cell_edges(SMALL, nominal))[low]
        rows = (data.size == size) & (data.volume >= low) & (data.volume < high)
        assert rows.sum() == count == 20, (size, low)
    assert len(data.y) == 8 * 20 and not data.y.any()


def test_a_cell_that_cannot_be_filled_is_reported_short_not_raised():
    short = DatasetConfig(
        sizes=("pet_250ml",),
        rows_per_size=10,
        volume_bins=(30.0, 31.0),
        rows_per_cell=20,
        max_draws_per_cell=7,
    )
    data, counts = cells(config=short)
    assert all(count <= 7 for count in counts.values())
    assert len(data.y) == sum(counts.values())


def test_the_calibration_rows_follow_the_seed_and_differ_from_the_training_rows():
    assert cells(seed=2)[0].X.tobytes() == cells(seed=2)[0].X.tobytes()
    train = {row.tobytes() for row in mixture(split="train").X}
    assert not train & {row.tobytes() for row in cells()[0].X}


def test_the_cell_at_the_threshold_holds_bottles_of_exactly_the_threshold():
    # The promise names 30 mL itself, the hardest volume to tell from empty: a threshold set on
    # bottles spread over 30 to 31 mL let 7 of 6000 bottles of exactly 30 mL through (2B.11).
    data, _ = cells()
    first = data.volume[data.volume < SMALL.volume_bins[1]]
    assert len(first) > 0 and set(first.tolist()) == {SMALL.volume_bins[0]}


def test_the_cells_over_the_threshold_still_spread_over_their_bins():
    data, _ = cells()
    later = data.volume[data.volume >= SMALL.volume_bins[1]]
    assert len(set(later.tolist())) > len(later) // 2
