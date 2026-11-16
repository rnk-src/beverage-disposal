"""What the camera reports about a bottle: sizes as intervals, the liquid level, cap and base.

Lengths are in metres. An interval is (low, high) and is meant to contain the true value; the
decision code treats it as a hard bound.
"""

import dataclasses
from dataclasses import dataclass

from beverage_disposal.core.bottles import Bottle, level_for_volume_m, total_height_m

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
