"""The training pipeline: it calibrates on its own split and leaves a bundle that loads."""

import dataclasses
import functools
import math

import numpy as np
import pytest
from test_dataset import FULLNESS, NAMES, NOISE, POLICY

from beverage_disposal.core.np_calibration import calibrate_cells, min_samples
from beverage_disposal.train.emptiness.bundle import load_bundle
from beverage_disposal.train.emptiness.dataset import DatasetConfig, generate_cells
from beverage_disposal.train.emptiness.fit import FitConfig
from beverage_disposal.train.emptiness.pipeline import train_bundle

# A looser alpha and delta than the real ones, so a small calibration set is enough.
LOOSE = dataclasses.replace(
    POLICY, config=dataclasses.replace(POLICY.config, alpha=0.05, delta=0.1)
)
BINS = (30.0, 35.0)
DATA = DatasetConfig(
    sizes=("pet_250ml", "pet_500ml"),
    rows_per_size=1500,
    volume_bins=BINS,
    rows_per_cell=60,
    max_draws_per_cell=20_000,
)
FIT = FitConfig(max_iter=30)
pytestmark = pytest.mark.medium


@functools.cache
def trained(folder):
    return train_bundle(LOOSE, FULLNESS.pad, NOISE, DATA, FIT, folder, seed=0)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return tmp_path_factory.mktemp("bundle")


def test_the_looser_figures_need_45_scores_per_cell():
    assert min_samples(0.05, 0.1) == 45 < DATA.rows_per_cell


def test_the_report_counts_the_rows_and_every_cell_is_filled(folder):
    _, report = trained(folder)
    assert 2000 < report.train_rows <= 3000
    assert report.cell_rows == {(size, low): 60 for size in DATA.sizes for low in BINS}
    assert report.empty_cells == [] and report.uncalibrated_sizes == []


def test_the_thresholds_come_from_the_calibration_split_and_not_the_training_rows(folder):
    bundle, _ = trained(folder)

    def taus(split):
        cells, _ = generate_cells(DATA, LOOSE, FULLNESS.pad, NOISE, split=split, seed=0)
        scores = bundle.model.scores(cells.X)
        return calibrate_cells(scores, cells.size, cells.volume, BINS, 0.05, 0.1)[0]

    assert bundle.tau_by_size == taus("calibration")
    assert bundle.tau_by_size != taus("train")
    assert all(0.0 < tau <= 1.0 for tau in bundle.tau_by_size.values())


def test_the_bundle_on_disk_loads_and_records_how_it_was_made(folder):
    bundle, _ = trained(folder)
    loaded = load_bundle(folder, NAMES)
    assert loaded.tau_by_size == bundle.tau_by_size and loaded.ranges == bundle.ranges
    assert loaded.meta["seed"] == 0 and (loaded.meta["alpha"], loaded.meta["delta"]) == (0.05, 0.1)
    assert loaded.meta["sizes"] == list(DATA.sizes)


def test_the_ranges_are_those_of_the_rows_the_model_was_fitted_on(folder):
    bundle, _ = trained(folder)
    width = NAMES.index("d_hi")
    # Only 250 and 500 mL bottles were drawn: no width near a 2 L bottle's is in range.
    assert bundle.ranges.hi[width] < 0.09
    wide = np.array(bundle.ranges.lo, dtype=float)
    wide[width] = 0.11
    assert not bundle.ranges.in_distribution(wide)


def test_a_size_with_a_cell_too_small_to_calibrate_can_never_be_overturned(tmp_path):
    short = dataclasses.replace(DATA, sizes=("pet_250ml",), rows_per_cell=20)
    bundle, report = train_bundle(LOOSE, FULLNESS.pad, NOISE, short, FIT, tmp_path, seed=0)
    assert report.uncalibrated_sizes == ["pet_250ml"]
    assert bundle.tau_by_size == {"pet_250ml": math.inf}
    assert load_bundle(tmp_path, NAMES).tau_by_size == {"pet_250ml": math.inf}
