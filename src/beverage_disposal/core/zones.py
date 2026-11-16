"""From a weight and the empty-mass bounds to an interval for the liquid, and its zone."""

from enum import Enum

from beverage_disposal.core.bounds import EmptyMassBounds
from beverage_disposal.core.fullness_config import RuleConfig


class Zone(Enum):
    SURELY_UNDER = "surely_under"
    AMBIGUOUS = "ambiguous"
    SURELY_OVER = "surely_over"


def liquid_bounds_ml(
    mass_kg: float, bounds: EmptyMassBounds, rule: RuleConfig
) -> tuple[float, float]:
    """The least and the most liquid, in mL, a bottle weighing `mass_kg` can hold.

    Most: the lightest bottle, the scale reading low by its margin, the lightest drink. Least:
    the heaviest bottle, the scale reading high, the heaviest drink; never below nothing.
    """
    grams = mass_kg * 1000.0
    most = (grams - bounds.min_kg * 1000.0 + rule.margin_g) / rule.min_density_g_per_ml
    least = (grams - bounds.max_kg * 1000.0 - rule.margin_g) / rule.max_density_g_per_ml
    return max(least, 0.0), most


def zone_of(v_lo: float, v_hi: float, threshold_ml: float) -> Zone:
    """Surely under when even the most is under the threshold; surely over when the least is not."""
    if v_hi < threshold_ml:
        return Zone.SURELY_UNDER
    if v_lo >= threshold_ml:
        return Zone.SURELY_OVER
    return Zone.AMBIGUOUS


def intersect_ml(
    first: tuple[float, float], second: tuple[float, float]
) -> tuple[float, float] | None:
    """The liquid both intervals allow, in mL; None when they share nothing."""
    lo = max(first[0], second[0])
    hi = min(first[1], second[1])
    if lo > hi:
        return None
    return (lo, hi)
