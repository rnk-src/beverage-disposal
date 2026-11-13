"""What a container is: its size, its empty weight and where those figures came from."""

from dataclasses import dataclass

PROVENANCE_KINDS = ("measured", "cited", "assumed")
# Water, the stand-in for any drink: one gram per millilitre. Juice and soda run to about 1.05.
LIQUID_KILOGRAMS_PER_ML = 1e-3


@dataclass(frozen=True)
class Provenance:
    source: str
    date: str
    kind: str

    def __post_init__(self):
        if self.kind not in PROVENANCE_KINDS:
            raise ValueError(f"kind must be one of {PROVENANCE_KINDS}, got {self.kind!r}")


@dataclass(frozen=True)
class ContainerSpec:
    name: str
    diameter: float
    height: float
    empty_mass: float
    grasp_height: float  # TCP height above the container's bottom when the jaws take it, metres
    friction: float
    provenance: Provenance


def total_mass(container: ContainerSpec, fill_ml: float) -> float:
    """Empty container plus `fill_ml` of liquid, in kilograms."""
    if fill_ml < 0:
        raise ValueError(f"fill must not be negative, got {fill_ml} mL")
    return container.empty_mass + fill_ml * LIQUID_KILOGRAMS_PER_ML
