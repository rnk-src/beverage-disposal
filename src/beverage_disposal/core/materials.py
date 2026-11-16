"""Material constants as data: every figure is a range with a source, one table per material."""

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path

from beverage_disposal.core.config import from_mapping
from beverage_disposal.core.containers import Provenance


@dataclass(frozen=True)
class Range:
    """A figure known only to lie between `lo` and `hi`, with where it came from."""

    lo: float
    hi: float
    provenance: Provenance

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"the low end {self.lo} is above the high end {self.hi}")


@dataclass(frozen=True)
class MaterialSpec:
    """What the empty-weight bounds need to know about a material. Lengths in mm, masses in g."""

    name: str
    density_g_cm3: Range
    t_side_mm: Range  # side wall thickness
    t_base_mm: Range  # base thickness
    finish_g: Range  # the threaded neck, which is moulded thick
    cap_g: Range  # a cap may be missing, so the low end is zero
    label_g: Range  # a label may be missing, so the low end is zero
    wet_g: Range  # water clinging to the outside, so the low end is zero


def load_materials(path: str | Path) -> dict[str, MaterialSpec]:
    """Read a materials file: one table per material, the table name being the material's name."""
    with open(path, "rb") as file:
        tables = tomllib.load(file)
    return {
        name: from_mapping(MaterialSpec, {"name": name, **table}, name)
        for name, table in tables.items()
    }


def with_side_wall(
    material: MaterialSpec, side_wall_mm: tuple[float, float]
) -> MaterialSpec | None:
    """The material with its side wall narrowed to a measured interval, in mm.

    The result's range is the overlap of the material's range and the measurement, so a measurement
    can only narrow. None when they do not overlap: the measurement is not of this material.
    """
    lo = max(material.t_side_mm.lo, side_wall_mm[0])
    hi = min(material.t_side_mm.hi, side_wall_mm[1])
    if lo > hi:
        return None
    narrowed = Range(lo, hi, material.t_side_mm.provenance)
    return dataclasses.replace(material, t_side_mm=narrowed)
