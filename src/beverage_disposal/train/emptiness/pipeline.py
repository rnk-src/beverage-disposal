"""From simulated bottles to a calibrated bundle on disk: draw, fit, calibrate, save."""

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beverage_disposal.core.classifier import FeatureRanges, ModelBundle
from beverage_disposal.core.fullness_config import PadSensorConfig
from beverage_disposal.core.np_calibration import calibrate_cells, min_samples
from beverage_disposal.core.perception import PerceptionNoise
from beverage_disposal.core.policy import EmptinessPolicy
from beverage_disposal.train.emptiness.bundle import save_bundle
from beverage_disposal.train.emptiness.dataset import DatasetConfig, generate, generate_cells
from beverage_disposal.train.emptiness.fit import FIT_ZONES, FitConfig, fit


@dataclass(frozen=True)
class TrainingReport:
    train_rows: int
    cell_rows: dict[tuple[str, float], int]  # calibration rows per size and volume cell
    empty_cells: list[tuple[str, float]]  # no ambiguous bottle was found: nothing to guard
    uncalibrated_sizes: list[str]  # a cell was short, so the size may never be overturned


def train_bundle(
    policy: EmptinessPolicy,
    pad: PadSensorConfig,
    noise: PerceptionNoise | None,
    data: DatasetConfig,
    fitting: FitConfig,
    folder: str | Path,
    *,
    seed: int,
) -> tuple[ModelBundle, TrainingReport]:
    """Fit on the train split, calibrate on the calibration split, and save the bundle.

    Alpha and delta come from the policy's config. A size with a cell too small to calibrate
    gets an infinite threshold: the classifier can then never bin a bottle of that size.
    """
    alpha, delta = policy.config.alpha, policy.config.delta
    train = generate(data, policy, pad, noise, split="train", seed=seed)
    model = fit(train, fitting)
    ranges = FeatureRanges.from_rows(train.X[np.isin(train.zone, FIT_ZONES)], train.names)

    cells, counts = generate_cells(data, policy, pad, noise, split="calibration", seed=seed)
    needed = min_samples(alpha, delta)
    short = sorted({size for (size, _), count in counts.items() if 0 < count < needed})
    usable = ~np.isin(cells.size, short)
    taus, empty = calibrate_cells(
        model.scores(cells.X[usable]) if usable.any() else np.array([]),
        cells.size[usable],
        cells.volume[usable],
        data.volume_bins,
        alpha,
        delta,
    )
    taus.update({size: math.inf for size in short})
    meta = {
        "seed": seed,
        "alpha": alpha,
        "delta": delta,
        "sizes": list(data.sizes),
        "volume_bins": list(data.volume_bins),
        "train_rows": len(train.y),
        "fit": dataclasses.asdict(fitting),
    }
    bundle = save_bundle(model, taus, ranges, train.names, meta, folder)
    empty_cells = [cell for cell, count in counts.items() if count == 0]
    return bundle, TrainingReport(len(train.y), counts, empty_cells, short)
