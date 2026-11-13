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
    torsional_friction: float  # resists twisting on a surface, metres (MuJoCo units)
    provenance: Provenance


def total_mass(container: ContainerSpec, fill_ml: float) -> float:
    """Empty container plus `fill_ml` of liquid, in kilograms."""
    if fill_ml < 0:
        raise ValueError(f"fill must not be negative, got {fill_ml} mL")
    return container.empty_mass + fill_ml * LIQUID_KILOGRAMS_PER_ML


@dataclass(frozen=True)
class ContainerRecord:
    """A container type as the registry holds it: its model spec plus what the rules need."""

    type_id: str
    spec: ContainerSpec
    lightest_empty_mass: float  # kg: the lightest plausible empty weight, the weight rule's tare
    weight_provenance: Provenance
    material: str
    cap_check_required: bool  # a camera must confirm the cap is on before the container is binned
    has_tab: bool

    def __post_init__(self):
        if not 0 < self.lightest_empty_mass <= self.spec.empty_mass:
            raise ValueError(
                f"{self.type_id}: the lightest empty mass {self.lightest_empty_mass} kg must be "
                f"positive and at most the typical empty mass {self.spec.empty_mass} kg"
            )
        if self.spec.name != self.type_id:
            raise ValueError(f"{self.type_id}: its spec is named {self.spec.name!r}")
