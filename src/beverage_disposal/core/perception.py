"""What the camera reports about a bottle: sizes as intervals, the liquid level, cap and base.

Lengths are in metres. An interval is (low, high) and is meant to contain the true value; the
decision code treats it as a hard bound.
"""

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beverage_disposal.core.bottles import BASE_STYLES, Bottle, level_for_volume_m, total_height_m
from beverage_disposal.core.config import from_mapping

Interval = tuple[float, float]


@dataclass(frozen=True)
class DimensionEstimate:
    body_diameter: Interval
    total_height: Interval
    shoulder_height: Interval
    neck_diameter: Interval
    neck_height: Interval

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            low, high = getattr(self, field.name)
            if low > high:
                raise ValueError(f"{field.name}: the low end {low} is above the high end {high}")


@dataclass(frozen=True)
class VisualEvidence:
    level_m: Interval | None  # the liquid's surface above the outside of the base; None if unseen
    cap_seen: bool | None
    base_style: str | None
    base_confidence: float


@dataclass(frozen=True)
class Perception:
    dims: DimensionEstimate
    visual: VisualEvidence
    confidence: float  # 0 to 1: how far the camera trusts this reading as a whole
    material: str


def exact_perception(bottle: Bottle, liquid_ml: float, material: str = "PET") -> Perception:
    """The truth as a perception: zero-width intervals, the true level, full confidence."""
    shape = bottle.shape
    level = level_for_volume_m(shape, liquid_ml)

    def exact(value: float) -> Interval:
        return (value, value)

    return Perception(
        dims=DimensionEstimate(
            body_diameter=exact(shape.body_diameter),
            total_height=exact(total_height_m(shape)),
            shoulder_height=exact(shape.shoulder_height),
            neck_diameter=exact(shape.neck_diameter),
            neck_height=exact(shape.neck_height),
        ),
        visual=VisualEvidence(
            level_m=exact(level),
            cap_seen=shape.cap_mass > 0,
            base_style=shape.base_style,
            base_confidence=1.0,
        ),
        confidence=1.0,
        material=material,
    )


@dataclass(frozen=True)
class PerceptionNoise:
    """The simulated camera's errors. Lengths in metres, rates as fractions."""

    dim_sigma_m: float
    dim_bias_max_m: float
    interval_k: float
    level_sigma_m: float
    label_band: tuple[float, float]
    level_dropout: float
    cap_miss: float
    base_style_error: float
    confidence_drop_crushed: float
    deformed_width_factor: float


def load_perception_noise(path: str | Path) -> PerceptionNoise:
    with open(path, "rb") as file:
        return from_mapping(PerceptionNoise, tomllib.load(file), "perception")


def perceive(
    bottle: Bottle,
    liquid_ml: float,
    noise: PerceptionNoise,
    rng: np.random.Generator,
    *,
    material: str = "PET",
    deformed: bool = False,
) -> Perception:
    """What the simulated camera reports for this bottle on one look.

    Every length gets its own random error and shares one calibration bias; its interval is wide
    enough to hold the truth through both. The level is missing on a dropout and behind a label.
    A deformed (crushed) bottle is seen with a wider width interval and a lower confidence.
    """
    shape = bottle.shape
    bias = rng.uniform(-noise.dim_bias_max_m, noise.dim_bias_max_m)
    half = noise.interval_k * noise.dim_sigma_m + noise.dim_bias_max_m

    def measured(true: float, widen: float = 1.0) -> Interval:
        estimate = true + bias + rng.normal(0.0, noise.dim_sigma_m)
        return (max(estimate - widen * half, 0.0), max(estimate + widen * half, 0.0))

    dims = DimensionEstimate(
        body_diameter=measured(
            shape.body_diameter, noise.deformed_width_factor if deformed else 1.0
        ),
        total_height=measured(total_height_m(shape)),
        shoulder_height=measured(shape.shoulder_height),
        neck_diameter=measured(shape.neck_diameter),
        neck_height=measured(shape.neck_height),
    )

    level = level_for_volume_m(shape, liquid_ml)
    estimate = level + rng.normal(0.0, noise.level_sigma_m)
    dropped = rng.random() < noise.level_dropout
    low, high = (edge * shape.body_height for edge in noise.label_band)
    hidden = shape.label_mass > 0 and low <= level <= high
    level_half = noise.interval_k * noise.level_sigma_m
    seen_level = (
        None
        if dropped or hidden
        else (max(estimate - level_half, 0.0), max(estimate + level_half, 0.0))
    )

    cap = shape.cap_mass > 0
    if rng.random() < noise.cap_miss:
        cap = not cap
    base = shape.base_style
    if rng.random() < noise.base_style_error:
        base = next(style for style in BASE_STYLES if style != base)

    return Perception(
        dims=dims,
        visual=VisualEvidence(
            level_m=seen_level,
            cap_seen=cap,
            base_style=base,
            base_confidence=1.0 - noise.base_style_error,
        ),
        confidence=1.0 - (noise.confidence_drop_crushed if deformed else 0.0),
        material=material,
    )
