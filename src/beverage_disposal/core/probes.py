"""Extra sensors beside the pad and the camera: a wall probe and a liquid probe.

A reading is an interval meant to contain the truth, and the decision code treats it as a hard
bound. The wall is in mm and the liquid in mL. The models are mechanism-agnostic: a probe is an
error size and a dropout rate, so hardware can be chosen later against the study's numbers.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beverage_disposal.core.bottles import Bottle
from beverage_disposal.core.config import from_mapping
from beverage_disposal.core.perception import Interval

M_TO_MM = 1000.0


@dataclass(frozen=True)
class ProbeReadings:
    """What the probes report for one bottle. A missing reading is None."""

    side_wall_mm: Interval | None = None
    liquid_ml: Interval | None = None

    def __post_init__(self) -> None:
        for name in ("side_wall_mm", "liquid_ml"):
            reading = getattr(self, name)
            if reading is not None and reading[0] > reading[1]:
                raise ValueError(
                    f"{name}: the low end {reading[0]} is above the high end {reading[1]}"
                )


NO_PROBES = ProbeReadings()


@dataclass(frozen=True)
class WallProbeModel:
    """A simulated side-wall thickness probe that does not squeeze the bottle (an optical or
    NIR gauge), so a cap makes no difference. Rates and errors are fractions."""

    rel_sigma: float  # random error of the estimate, one standard deviation, relative
    interval_k: float  # the reading spans k sigma either way
    nonuniformity: float  # extra relative half-width: a real wall varies along the bottle
    dropout: float  # how often there is no reading

    @property
    def half_width(self) -> float:
        return self.interval_k * self.rel_sigma + self.nonuniformity

    def __post_init__(self) -> None:
        if self.rel_sigma < 0 or self.nonuniformity < 0:
            raise ValueError("the wall probe's errors cannot be negative")
        if not 0 <= self.dropout <= 1:
            raise ValueError("the dropout rate must lie between 0 and 1")
        if self.half_width >= 1:
            raise ValueError(f"the half width {self.half_width} must be under 1")


@dataclass(frozen=True)
class LiquidProbeModel:
    """A simulated probe of the liquid itself. Volumes in mL."""

    sigma_ml: float  # additive random error, one standard deviation
    rel_sigma: float  # multiplicative random error, one standard deviation
    interval_k: float
    dropout: float

    def __post_init__(self) -> None:
        if self.sigma_ml < 0 or self.rel_sigma < 0:
            raise ValueError("the liquid probe's errors cannot be negative")
        if not 0 <= self.dropout <= 1:
            raise ValueError("the dropout rate must lie between 0 and 1")
        if self.interval_k * self.rel_sigma >= 1:
            raise ValueError("the relative slack k * rel_sigma must be under 1")


@dataclass(frozen=True)
class ProbeModels:
    """Which probes are fitted; None means that probe is absent."""

    wall: WallProbeModel | None = None
    liquid: LiquidProbeModel | None = None


def load_probe_models(path: str | Path) -> ProbeModels:
    """Read both probe models from a file with a [wall] and a [liquid] table."""
    with open(path, "rb") as file:
        data = tomllib.load(file)
    unknown = sorted(set(data) - {"wall", "liquid"})
    if unknown:
        raise ValueError(f"probes: unknown key(s) {unknown}")
    for table in ("wall", "liquid"):
        if table not in data:
            raise ValueError(f"probes: missing table [{table}]")
    return ProbeModels(
        wall=from_mapping(WallProbeModel, data["wall"], "probes.wall"),
        liquid=from_mapping(LiquidProbeModel, data["liquid"], "probes.liquid"),
    )


def probe(
    bottle: Bottle,
    liquid_ml: float,
    models: ProbeModels,
    rng: np.random.Generator,
    *,
    liquid_gain: float = 1.0,
) -> ProbeReadings:
    """What the fitted probes report for this bottle on one try.

    `liquid_gain` scales what the liquid probe sees before its own errors: under 1 is a probe that
    reads low (a domed base holding the liquid away from it), and 0 is contents it cannot see at
    all (frozen or slushed). Both are the unsafe direction.
    """
    # Every draw happens whether or not its probe is fitted, so fitting one never moves the other.
    wall_dropped = rng.random()
    wall_error = rng.normal()
    liquid_dropped = rng.random()
    liquid_rel = rng.normal()
    liquid_add = rng.normal()

    wall = None
    if models.wall is not None and wall_dropped >= models.wall.dropout:
        wall_model = models.wall
        true_mm = bottle.shape.t_side * M_TO_MM
        estimate = true_mm * (1 + wall_model.rel_sigma * wall_error)
        if estimate > 0:
            h = wall_model.half_width
            # The truth is inside exactly when the relative error is within h.
            wall = (estimate / (1 + h), estimate / (1 - h))

    liquid = None
    if models.liquid is not None and liquid_dropped >= models.liquid.dropout:
        liquid_model = models.liquid
        estimate = (
            liquid_ml * liquid_gain * (1 + liquid_model.rel_sigma * liquid_rel)
            + liquid_model.sigma_ml * liquid_add
        )
        r = liquid_model.interval_k * liquid_model.rel_sigma
        a = liquid_model.interval_k * liquid_model.sigma_ml
        liquid = (max(estimate - a, 0.0) / (1 + r), max(estimate + a, 0.0) / (1 - r))

    return ProbeReadings(wall, liquid)
