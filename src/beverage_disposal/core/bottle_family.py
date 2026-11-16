"""PET bottles as a family: ranges of real shapes per nominal size, and a seeded sampler.

Lengths are in metres and masses in kilograms here; the material table is in millimetres and grams
and is converted once, where a bottle is drawn.
"""

import dataclasses
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beverage_disposal.core.bottles import (
    BASE_STYLES,
    M3_TO_ML,
    Bottle,
    BottleShape,
    capacity_ml,
    total_height_m,
)
from beverage_disposal.core.config import SceneConfig, from_mapping
from beverage_disposal.core.containers import ContainerSpec, Provenance
from beverage_disposal.core.materials import MaterialSpec
from beverage_disposal.core.perception import DimensionEstimate

GATE_STATUSES = ("untested", "passed", "failed")
GRASP_MODES = ("neck", "body")
CAP_PRESENT_P = 0.9  # assumed: most bottles come with the cap screwed on
LABEL_PRESENT_P = 0.8  # assumed: some labels are peeled off
MAX_DRAWS = 1000
MM = 1e-3  # metres per millimetre
GRAM = 1e-3  # kilograms per gram
PROBE_RISE = 0.001  # a body this much taller than its base, used only to measure the capacity


@dataclass(frozen=True)
class Span:
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"the low end {self.lo} is above the high end {self.hi}")


@dataclass(frozen=True)
class GateRecord:
    """What the 2B.3 neck-grasp gate found for a size."""

    status: str
    date: str
    note: str

    def __post_init__(self) -> None:
        if self.status not in GATE_STATUSES:
            raise ValueError(f"gate status must be one of {GATE_STATUSES}, got {self.status!r}")


@dataclass(frozen=True)
class SizeClass:
    """The ranges one nominal size's bottles are drawn from, with one source for the table."""

    name: str
    nominal_ml: float
    body_diameter: Span
    body_height: Span  # the range the solved body height must fall in; it is not drawn
    shoulder_height: Span
    neck_diameter: Span
    neck_height: Span
    support_ring_diameter: Span
    support_ring_drop: Span
    overfill: Span  # brim capacity over the nominal size
    base_styles: tuple[str, ...]
    grasp_mode: str
    # Metres from the grasp point up to the ring's underside (neck mode) or the shoulder (body).
    grasp_below_ring: float
    grasp_pitch_deg: float
    gate: GateRecord
    provenance: Provenance

    def __post_init__(self) -> None:
        if self.grasp_mode not in GRASP_MODES:
            raise ValueError(f"grasp_mode must be one of {GRASP_MODES}, got {self.grasp_mode!r}")
        if not self.base_styles or not set(self.base_styles) <= set(BASE_STYLES):
            raise ValueError(f"base_styles must be a non-empty part of {BASE_STYLES}")


@dataclass(frozen=True)
class BottleFamily:
    classes: dict[str, SizeClass]

    def supported(self) -> tuple[str, ...]:
        """The sizes that passed the neck-grasp gate, in file order."""
        return tuple(name for name, size in self.classes.items() if size.gate.status == "passed")


def load_family(path: str | Path) -> BottleFamily:
    """Read a family file: one table per size class, the table name being the class's name."""
    with open(path, "rb") as file:
        tables = tomllib.load(file)
    return BottleFamily(
        {
            name: from_mapping(SizeClass, {"name": name, **table}, name)
            for name, table in tables.items()
        }
    )


def sample_bottle(
    family: BottleFamily, size: str, material: MaterialSpec, rng: np.random.Generator
) -> Bottle:
    """A bottle drawn from the size's ranges; the same generator state gives the same bottle."""
    if size not in family.classes:
        raise ValueError(f"unknown size class {size!r}")
    spec = family.classes[size]
    for _ in range(MAX_DRAWS):
        bottle = _draw(spec, material, rng)
        if bottle is not None:
            return bottle
    raise ValueError(
        f"{size}: no draw gave a body height in {spec.body_height} after {MAX_DRAWS} tries"
    )


def _draw(spec: SizeClass, material: MaterialSpec, rng: np.random.Generator) -> Bottle | None:
    def uniform(span) -> float:
        return float(rng.uniform(span.lo, span.hi))

    diameter = uniform(spec.body_diameter)
    shoulder = uniform(spec.shoulder_height)
    neck_diameter = uniform(spec.neck_diameter)
    neck_height = uniform(spec.neck_height)
    ring_diameter = uniform(spec.support_ring_diameter)
    ring_drop = uniform(spec.support_ring_drop)
    overfill = uniform(spec.overfill)
    base_style = spec.base_styles[int(rng.integers(len(spec.base_styles)))]
    t_side = uniform(material.t_side_mm) * MM
    t_base = uniform(material.t_base_mm) * MM
    finish = uniform(material.finish_g) * GRAM
    cap = uniform(material.cap_g) * GRAM if rng.random() < CAP_PRESENT_P else 0.0
    label = uniform(material.label_g) * GRAM if rng.random() < LABEL_PRESENT_P else 0.0
    density = uniform(material.density_g_cm3)

    def shape_with(body_height: float) -> BottleShape:
        return BottleShape(
            body_diameter=diameter,
            body_height=body_height,
            shoulder_height=shoulder,
            neck_diameter=neck_diameter,
            neck_height=neck_height,
            base_style=base_style,
            t_side=t_side,
            t_base=t_base,
            finish_mass=finish,
            cap_mass=cap,
            label_mass=label,
            support_ring_diameter=ring_diameter,
            support_ring_drop=ring_drop,
        )

    # The capacity is linear in the body height, so one probe shape gives the exact height.
    probe = shape_with(t_base + PROBE_RISE)
    inner_area_ml_per_m = math.pi * (diameter / 2 - t_side) ** 2 * M3_TO_ML
    wanted = spec.nominal_ml * overfill
    height = probe.body_height + (wanted - capacity_ml(probe)) / inner_area_ml_per_m
    if not spec.body_height.lo <= height <= spec.body_height.hi:
        return None
    return Bottle(shape_with(height), density, spec.name)


def container_spec_for(
    bottle: Bottle, size: SizeClass, *, friction: float, torsional_friction: float
) -> ContainerSpec:
    """The bottle as the jaws and the task tests see it: width and height of the grasp included.

    No bottle's friction is measured, so the caller says which figures to use.
    """
    shape = bottle.shape
    if size.grasp_mode == "neck":
        if shape.support_ring_diameter <= 0:
            raise ValueError(f"{size.name}: a neck grasp needs a support ring to stay below")
        grasp_height = total_height_m(shape) - shape.support_ring_drop - size.grasp_below_ring
        grasp_diameter = shape.neck_diameter
    else:
        grasp_height = shape.body_height - size.grasp_below_ring
        grasp_diameter = shape.body_diameter
    return ContainerSpec(
        name=size.name,
        diameter=shape.body_diameter,
        grasp_diameter=grasp_diameter,
        height=total_height_m(shape),
        empty_mass=bottle.empty_mass_kg,
        grasp_height=grasp_height,
        friction=friction,
        torsional_friction=torsional_friction,
        provenance=size.provenance,
    )


def scene_for_bottle(
    scene: SceneConfig,
    bottle: Bottle,
    size: SizeClass,
    liquid_ml: float,
    density_g_ml: float = 1.0,
) -> SceneConfig:
    """`scene` with this bottle as its container, `liquid_ml` of a drink of that density in it.

    The bottle takes the friction of the scene's current container (the can) until it is measured.
    """
    can = scene.container
    clearance = scene.grasp.clearance
    if size.grasp_mode == "neck":
        # The jaw comes down past the ring before it closes on the narrower neck under it.
        clearance += (bottle.shape.support_ring_diameter - bottle.shape.neck_diameter) / 2
    return dataclasses.replace(
        scene,
        container=container_spec_for(
            bottle, size, friction=can.friction, torsional_friction=can.torsional_friction
        ),
        bottle=bottle,
        fill_ml=liquid_ml,
        fill_density_g_ml=density_g_ml,
        grasp=dataclasses.replace(scene.grasp, pitch_deg=size.grasp_pitch_deg, clearance=clearance),
    )


def classes_matching(family: BottleFamily, dims: DimensionEstimate) -> tuple[str, ...]:
    """The size classes a measured bottle could belong to, in file order.

    A class matches when the measured width overlaps its width range and the measured height
    overlaps the heights its body, shoulder and neck ranges can add up to.
    """

    def overlap(interval, low: float, high: float) -> bool:
        return interval[0] <= high and interval[1] >= low

    found = []
    for name, size in family.classes.items():
        parts = (size.body_height, size.shoulder_height, size.neck_height)
        shortest, tallest = sum(part.lo for part in parts), sum(part.hi for part in parts)
        width = size.body_diameter
        if overlap(dims.body_diameter, width.lo, width.hi) and overlap(
            dims.total_height, shortest, tallest
        ):
            found.append(name)
    return tuple(found)
