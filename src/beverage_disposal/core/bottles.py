"""A plastic bottle as a shape: areas, shell mass, capacity and where a liquid's level sits.

The shape is a body cylinder, a conical shoulder and a neck cylinder, standing on a flat base.
Lengths are in metres, masses in kilograms. A level is a height above the outside of the base,
which is what a camera sees; the liquid starts at the inside of the base.
"""

import math
from dataclasses import dataclass

BASE_STYLES = ("flat", "petaloid")
M3_TO_ML = 1e6
G_CM3_TO_KG_M3 = 1000.0
BISECTIONS = 100  # halves the bracket to far below any float difference in a level


@dataclass(frozen=True)
class BottleShape:
    body_diameter: float
    body_height: float
    shoulder_height: float
    neck_diameter: float
    neck_height: float
    base_style: str
    t_side: float  # wall thickness of the body and the shoulder
    t_base: float
    finish_mass: float  # the threaded neck, moulded thick, taken as one lump
    cap_mass: float
    label_mass: float

    def __post_init__(self) -> None:
        for name in ("body_diameter", "body_height", "neck_diameter", "neck_height"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in (
            "shoulder_height",
            "t_side",
            "t_base",
            "finish_mass",
            "cap_mass",
            "label_mass",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative, got {getattr(self, name)}")
        if self.neck_diameter > self.body_diameter:
            raise ValueError("the neck must not be wider than the body")
        if self.base_style not in BASE_STYLES:
            raise ValueError(f"base_style must be one of {BASE_STYLES}, got {self.base_style!r}")
        if self.t_side >= self.neck_diameter / 2:
            raise ValueError("the wall must be thinner than the neck's radius")
        if self.t_base >= self.body_height:
            raise ValueError("the base must be thinner than the body is tall")


@dataclass(frozen=True)
class Areas:
    """Outside surface areas in square metres."""

    side: float
    base: float
    shoulder: float


def total_height_m(shape: BottleShape) -> float:
    return shape.body_height + shape.shoulder_height + shape.neck_height


def areas(shape: BottleShape) -> Areas:
    big, small = shape.body_diameter / 2, shape.neck_diameter / 2
    slant = math.hypot(shape.shoulder_height, big - small)
    return Areas(
        side=math.pi * shape.body_diameter * shape.body_height,
        base=math.pi * big**2,
        shoulder=math.pi * (big + small) * slant,
    )


def shell_mass_kg(shape: BottleShape, density_g_cm3: float) -> float:
    """Mass of the walls and base alone: each outside area times its thickness times the density."""
    found = areas(shape)
    volume = (found.side + found.shoulder) * shape.t_side + found.base * shape.t_base
    return density_g_cm3 * G_CM3_TO_KG_M3 * volume


def empty_mass_kg(shape: BottleShape, density_g_cm3: float) -> float:
    """The shell plus the neck, the cap and the label."""
    return shell_mass_kg(shape, density_g_cm3) + (
        shape.finish_mass + shape.cap_mass + shape.label_mass
    )


def volume_below_ml(shape: BottleShape, level_m: float) -> float:
    """Liquid held below a level, in mL: zero at or under the floor, the capacity above the top."""
    floor = shape.t_base
    if level_m <= floor:
        return 0.0
    rise = min(level_m, total_height_m(shape)) - floor
    big = shape.body_diameter / 2 - shape.t_side
    small = shape.neck_diameter / 2 - shape.t_side
    volume = 0.0

    up = min(rise, shape.body_height - floor)
    volume += math.pi * big**2 * up
    rise -= up

    if shape.shoulder_height > 0:
        up = min(rise, shape.shoulder_height)
        radius = big + (small - big) * up / shape.shoulder_height
        volume += math.pi * up / 3 * (big**2 + big * radius + radius**2)
        rise -= up

    volume += math.pi * small**2 * min(rise, shape.neck_height)
    return volume * M3_TO_ML


def capacity_ml(shape: BottleShape) -> float:
    """What the bottle holds when filled to the brim of the neck."""
    return volume_below_ml(shape, total_height_m(shape))


def level_for_volume_m(shape: BottleShape, volume_ml: float) -> float:
    """The height above the outside of the base at which `volume_ml` of liquid stands."""
    capacity = capacity_ml(shape)
    if volume_ml < 0 or volume_ml > capacity * (1 + 1e-12):
        raise ValueError(f"volume {volume_ml} mL is outside 0 to {capacity} mL")
    low, high = shape.t_base, total_height_m(shape)
    for _ in range(BISECTIONS):
        middle = (low + high) / 2
        if volume_below_ml(shape, middle) < volume_ml:
            low = middle
        else:
            high = middle
    return (low + high) / 2
