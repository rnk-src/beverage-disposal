"""The lightest and heaviest a plastic bottle of a measured size can be when empty.

No database of empty weights: the shell's surface area times a wall-thickness range and the
plastic's density, plus the neck finish, and for the heavy end a cap, a label and water on the
outside. Every step takes the conservative end of each interval.
"""

import math
from dataclasses import dataclass

from beverage_disposal.core.bottles import G_CM3_TO_KG_M3
from beverage_disposal.core.materials import MaterialSpec
from beverage_disposal.core.perception import DimensionEstimate, Interval

MM = 1e-3  # metres per millimetre
GRAM = 1e-3  # kilograms per gram


@dataclass(frozen=True)
class EmptyMassBounds:
    min_kg: float
    max_kg: float


def shoulder_area_bounds_m2(big: Interval, small: Interval, height: Interval) -> Interval:
    """Bounds on a conical shoulder's area from intervals for its two radii and its height.

    The area pi (R + r) sqrt(s^2 + (R - r)^2) is not monotone in the small radius, so the ends of
    the intervals cannot simply be plugged in. Below, it is at least the flat ring pi (R^2 - r^2)
    and at least pi (R + r) s; above, the slant is at most s + (R - r).
    """
    low = max(
        math.pi * (big[0] ** 2 - small[1] ** 2),
        math.pi * (big[0] + small[0]) * height[0],
        0.0,
    )
    high = math.pi * (big[1] + small[1]) * (height[1] + big[1] - small[0])
    return low, high


def empty_mass_bounds(dims: DimensionEstimate, material: MaterialSpec) -> EmptyMassBounds:
    """The lightest dry bottle and the heaviest capped, labelled, wet one these sizes allow."""
    diameter, neck = dims.body_diameter, dims.neck_diameter
    big = (diameter[0] / 2, diameter[1] / 2)
    small = (neck[0] / 2, neck[1] / 2)
    # The body is what is left of the height under the shoulder and the neck.
    body_lo = dims.total_height[0] - dims.shoulder_height[1] - dims.neck_height[1]
    body_hi = dims.total_height[1] - dims.shoulder_height[0] - dims.neck_height[0]
    if body_hi <= 0:
        raise ValueError("the height leaves no room for a body under the shoulder and the neck")
    side = (math.pi * diameter[0] * max(body_lo, 0.0), math.pi * diameter[1] * body_hi)
    base = (math.pi * big[0] ** 2, math.pi * big[1] ** 2)
    shoulder = shoulder_area_bounds_m2(big, small, dims.shoulder_height)

    def shell_kg(end: int, density: float, t_side_mm: float, t_base_mm: float) -> float:
        volume = (side[end] + shoulder[end]) * t_side_mm * MM + base[end] * t_base_mm * MM
        return density * G_CM3_TO_KG_M3 * volume

    m = material
    lightest = (
        shell_kg(0, m.density_g_cm3.lo, m.t_side_mm.lo, m.t_base_mm.lo) + m.finish_g.lo * GRAM
    )
    extras_g = m.finish_g.hi + m.cap_g.hi + m.label_g.hi + m.wet_g.hi
    heaviest = shell_kg(1, m.density_g_cm3.hi, m.t_side_mm.hi, m.t_base_mm.hi) + extras_g * GRAM
    return EmptyMassBounds(lightest, heaviest)
