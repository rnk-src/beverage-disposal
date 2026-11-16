"""Simulated bottles as labelled rows for the emptiness classifier.

A row is one bottle on the pad: its feature vector, whether it truly holds under the threshold,
and the zone its weight put it in. Vetoed bottles never reach the classifier and are left out.
"""

from dataclasses import dataclass

import numpy as np

from beverage_disposal.core.emptiness_eval import Case, draw_case
from beverage_disposal.core.features import feature_names, feature_vector
from beverage_disposal.core.fullness_config import PadSensorConfig
from beverage_disposal.core.pad import PadSensor
from beverage_disposal.core.perception import PerceptionNoise
from beverage_disposal.core.policy import EmptinessPolicy
from beverage_disposal.core.zones import Zone

# Each split draws from its own stream, so no bottle is in two of them.
SPLIT_STREAMS = {"train": 0, "calibration": 1, "test": 2, "acceptance": 3}
MIXTURE_STREAM, CELL_STREAM = 0, 1  # a mixture and the cells of one split do not share bottles
KG_TO_ML_OF_WATER = 1000.0
UNIFORM_TOP_ML = 60.0  # the plain part of the mixture covers 0 to this, around the threshold


@dataclass(frozen=True)
class DatasetConfig:
    sizes: tuple[str, ...]
    rows_per_size: int  # bottles drawn per size for a mixture; vetoed ones are dropped
    volume_bins: tuple[float, ...]  # mL: lower edges of the volume cells at and over the threshold
    rows_per_cell: int = 598  # ambiguous rows wanted per size and volume cell
    max_draws_per_cell: int = 200_000
    near_threshold_weight: float = 0.5  # the share of a mixture drawn from 0 to 60 mL
    payload_kg: float = 0.6


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray  # (n, features) float64
    y: np.ndarray  # (n,) bool: the bottle holds under the threshold
    volume: np.ndarray  # (n,) mL, the truth
    size: np.ndarray  # (n,) the size class each bottle was drawn from
    zone: np.ndarray  # (n,) the zone's name
    names: tuple[str, ...]
    seed: tuple[int, int]  # the base seed and the split's stream


def cell_edges(config: DatasetConfig, size_nominal_ml: float) -> list[tuple[float, float]]:
    """The volume cells of a size as (low, high) in mL; the last one ends at what it can hold."""
    top = min(size_nominal_ml, config.payload_kg * KG_TO_ML_OF_WATER)
    edges = [*config.volume_bins, top]
    return list(zip(edges[:-1], edges[1:], strict=True))


class _Rows:
    """Rows being collected, turned into a Dataset at the end."""

    def __init__(self, policy: EmptinessPolicy):
        self._policy = policy
        self._columns: tuple[list, ...] = ([], [], [], [], [])

    def add(self, case: Case, size: str, only_zone: Zone | None = None) -> bool:
        """Add the bottle unless it is vetoed or outside `only_zone`; say whether it was added."""
        policy = self._policy
        found = policy.assess(case.perception, case.weighing)
        if found.veto is not None or (only_zone is not None and found.zone is not only_zone):
            return False
        assert found.bounds is not None and found.v_bounds is not None and found.zone is not None
        features = feature_vector(
            case.perception,
            case.weighing,
            found.bounds,
            found.v_bounds,
            policy.family,
            policy.material,
        )
        row = (
            features,
            case.volume_ml < policy.rule.threshold_ml,
            case.volume_ml,
            size,
            found.zone.value,
        )
        for column, value in zip(self._columns, row, strict=True):
            column.append(value)
        return True

    def dataset(self, seed: tuple[int, int]) -> Dataset:
        features, labels, volumes, sizes, zones = self._columns
        names = feature_names(self._policy.family)
        return Dataset(
            X=np.array(features, dtype=np.float64).reshape(len(labels), len(names)),
            y=np.array(labels, dtype=bool),
            volume=np.array(volumes, dtype=np.float64),
            size=np.array(sizes, dtype=str),
            zone=np.array(zones, dtype=str),
            names=names,
            seed=seed,
        )


def _stream(split: str, seed: int) -> tuple[int, int]:
    if split not in SPLIT_STREAMS:
        raise ValueError(f"split must be one of {tuple(SPLIT_STREAMS)}, got {split!r}")
    return (seed, SPLIT_STREAMS[split])


def generate(
    config: DatasetConfig,
    policy: EmptinessPolicy,
    pad: PadSensorConfig,
    noise: PerceptionNoise | None,
    *,
    split: str,
    seed: int,
) -> Dataset:
    """A mixture for fitting or testing: volumes around the threshold and across the cells."""
    stream = _stream(split, seed)
    rng = np.random.default_rng([*stream, MIXTURE_STREAM])
    sensor = PadSensor(pad, rng)
    rows = _Rows(policy)
    for size in config.sizes:
        edges = cell_edges(config, policy.family.classes[size].nominal_ml)
        for _ in range(config.rows_per_size):
            if rng.random() < config.near_threshold_weight:
                volume = rng.uniform(0.0, UNIFORM_TOP_ML)
            else:
                low, high = edges[int(rng.integers(len(edges)))]
                volume = rng.uniform(low, high)
            rows.add(draw_case(policy, sensor, noise, size, float(volume), rng), size)
    return rows.dataset(stream)


def generate_cells(
    config: DatasetConfig,
    policy: EmptinessPolicy,
    pad: PadSensorConfig,
    noise: PerceptionNoise | None,
    *,
    split: str,
    seed: int,
) -> tuple[Dataset, dict[tuple[str, float], int]]:
    """Ambiguous-zone rows per size and volume cell, for calibration, and each cell's row count.

    A cell is filled to `rows_per_cell` rows or until `max_draws_per_cell` bottles were drawn;
    a cell that stays short or empty is reported by its count, not raised.
    """
    stream = _stream(split, seed)
    rng = np.random.default_rng([*stream, CELL_STREAM])
    sensor = PadSensor(pad, rng)
    rows = _Rows(policy)
    counts = {}
    for size in config.sizes:
        for low, high in cell_edges(config, policy.family.classes[size].nominal_ml):
            kept = 0
            for _ in range(config.max_draws_per_cell):
                if kept == config.rows_per_cell:
                    break
                volume = float(rng.uniform(low, high))
                case = draw_case(policy, sensor, noise, size, volume, rng)
                kept += rows.add(case, size, only_zone=Zone.AMBIGUOUS)
            counts[(size, low)] = kept
    return rows.dataset(stream), counts
