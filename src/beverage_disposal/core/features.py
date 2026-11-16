"""The classifier's inputs: one fixed-order vector per bottle, from what the camera and pad report.

Nothing here knows the truth about a bottle: every number comes from the perception, the weighing
and the bounds worked out from them.
"""

import math

import numpy as np

from beverage_disposal.core.bottle_family import BottleFamily, classes_matching
from beverage_disposal.core.bottles import BottleShape, volume_below_ml
from beverage_disposal.core.bounds import EmptyMassBounds
from beverage_disposal.core.materials import MaterialSpec
from beverage_disposal.core.pad import Weighing
from beverage_disposal.core.perception import Perception

BASE_FEATURES = (
    "d_lo",
    "d_hi",
    "h_lo",
    "h_hi",
    "s_lo",
    "s_hi",
    "neck_lo",
    "neck_hi",
    "w_g",
    "emin_g",
    "emax_g",
    "v_lo",
    "v_hi",
    "level_seen",
    "level_v_lo",
    "level_v_hi",
    "cap_seen",
    "base_petaloid",
    "base_confidence",
    "confidence",
)
# The features that may be missing (NaN): the level's volume when the level was not seen.
MAY_BE_MISSING = ("level_v_lo", "level_v_hi")
MM = 1e-3  # metres per millimetre
KG_TO_G = 1000.0


def feature_names(family: BottleFamily) -> tuple[str, ...]:
    """The base features, then one `size_<class>` flag per size class in file order."""
    return BASE_FEATURES + tuple(f"size_{name}" for name in family.classes)


def _middle(interval: tuple[float, float]) -> float:
    return (interval[0] + interval[1]) / 2


def level_volume_bounds_ml(
    perception: Perception, material: MaterialSpec
) -> tuple[float, float] | None:
    """The liquid a seen level stands for, in mL, at the two ends of the level's interval.

    Worked out on a nominal bottle: the middle of each measured length and of the material's
    wall and base thickness. None when the level was not seen, or when the measured lengths do
    not make a bottle.
    """
    level = perception.visual.level_m
    if level is None:
        return None
    dims = perception.dims
    shoulder, neck = _middle(dims.shoulder_height), _middle(dims.neck_height)
    try:
        nominal = BottleShape(
            body_diameter=_middle(dims.body_diameter),
            body_height=_middle(dims.total_height) - shoulder - neck,
            shoulder_height=shoulder,
            neck_diameter=_middle(dims.neck_diameter),
            neck_height=neck,
            base_style="flat",
            t_side=_middle((material.t_side_mm.lo, material.t_side_mm.hi)) * MM,
            t_base=_middle((material.t_base_mm.lo, material.t_base_mm.hi)) * MM,
            finish_mass=0.0,
            cap_mass=0.0,
            label_mass=0.0,
        )
    except ValueError:
        return None
    return volume_below_ml(nominal, level[0]), volume_below_ml(nominal, level[1])


def _flag(value: bool | None) -> float:
    return math.nan if value is None else float(value)


def feature_vector(
    perception: Perception,
    weighing: Weighing,
    bounds: EmptyMassBounds,
    v_bounds: tuple[float, float],
    family: BottleFamily,
    material: MaterialSpec,
) -> np.ndarray:
    """One bottle as the classifier sees it, in the order of `feature_names`."""
    dims, visual = perception.dims, perception.visual
    level = level_volume_bounds_ml(perception, material)
    base = visual.base_style
    matches = classes_matching(family, dims)
    values = [
        *dims.body_diameter,
        *dims.total_height,
        *dims.shoulder_height,
        *dims.neck_diameter,
        weighing.mass_kg * KG_TO_G,
        bounds.min_kg * KG_TO_G,
        bounds.max_kg * KG_TO_G,
        *v_bounds,
        float(visual.level_m is not None),
        *(level or (math.nan, math.nan)),
        _flag(visual.cap_seen),
        _flag(None if base is None else base == "petaloid"),
        visual.base_confidence,
        perception.confidence,
        *(float(name in matches) for name in family.classes),
    ]
    return np.array(values, dtype=np.float64)
