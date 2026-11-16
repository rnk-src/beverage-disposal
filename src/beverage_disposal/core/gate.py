"""The neck-grasp gate: what is measured on a recorded carry, and the bar a size must clear."""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beverage_disposal.core.config import from_mapping
from beverage_disposal.core.trajectory import Trajectory

# Rows the tilt must stay in the band at the end of a hold to count as settled (0.25 s).
MIN_QUIET_ROWS = 5


@dataclass(frozen=True)
class GateBar:
    max_slip_m: float
    max_swing_settle_s: float
    required_places: int
    swing_band_rad: float
    seeds: int


@dataclass(frozen=True)
class GateRow:
    """One size at one fill: a slip, a settle time and a place result per seeded run."""

    size: str
    fill: str
    slips: tuple[float, ...]
    settle_times: tuple[float, ...]
    places_upright: tuple[bool, ...]


def grip_slip_m(trajectory: Trajectory, grip_index: int, release_index: int) -> float:
    """How far the container moved in the grip between the settled grip and the release, metres.

    Read on the row the grip settled and on the last row before the release; infinite when the
    run never got that far.
    """
    last = release_index - 1
    if not 0 <= grip_index <= last < len(trajectory.time):
        return math.inf
    offset = trajectory.container - trajectory.tip
    return float(np.linalg.norm(offset[last] - offset[grip_index]))


def swing_settle_s(
    trajectory: Trajectory, carry_end_index: int, band_rad: float, end_index: int | None = None
) -> float:
    """Seconds after the carry until the tilt stays within `band_rad` of where it ends.

    The rows from the end of the carry up to `end_index` (the release; the end of the record when
    not given) are the hold. Infinite when the tilt is still outside the band in its last rows.
    """
    end = len(trajectory.time) if end_index is None else end_index
    if not 0 <= carry_end_index < end <= len(trajectory.time):
        return math.inf
    tilt = np.arccos(np.clip(trajectory.upright[carry_end_index:end], -1.0, 1.0))
    outside = np.flatnonzero(np.abs(tilt - tilt[-1]) > band_rad)
    if len(outside) == 0:
        return 0.0
    settled = int(outside[-1]) + 1
    if len(tilt) - settled < MIN_QUIET_ROWS:
        return math.inf
    times = trajectory.time[carry_end_index:end]
    return float(times[settled] - times[0])


def passes(row: GateRow, bar: GateBar) -> bool:
    """Whether every run of the row was placed upright within the slip and swing limits."""
    return (
        sum(row.places_upright) >= bar.required_places
        and len(row.slips) >= bar.required_places
        and max(row.slips) <= bar.max_slip_m
        and max(row.settle_times) <= bar.max_swing_settle_s
    )


def load_gate_bar(path: str | Path) -> GateBar:
    with open(path, "rb") as file:
        return from_mapping(GateBar, tomllib.load(file), "gate")
