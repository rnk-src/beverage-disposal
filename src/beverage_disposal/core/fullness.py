"""The weight rule: bin only when sure the container holds under the threshold, else set aside."""

from dataclasses import dataclass

import numpy as np

from beverage_disposal.core.containers import ContainerRecord
from beverage_disposal.core.fullness_config import RuleConfig
from beverage_disposal.core.pad import Weighing

BIN = "bin"
SET_ASIDE = "set_aside"


@dataclass(frozen=True)
class Verdict:
    action: str  # BIN or SET_ASIDE
    reason: str
    net_g: float | None  # weight above the lightest empty container, when it was worked out


def bin_mask(net_g: np.ndarray, rule: RuleConfig) -> np.ndarray:
    """Which net weights (grams) are binned: not lighter than the margin allows, under the limit."""
    return (net_g >= -rule.margin_g) & (net_g < rule.limit_g)


def decide(
    rule: RuleConfig,
    record: ContainerRecord | None,
    weighing: Weighing,
    cap_confirmed: bool | None = None,
) -> Verdict:
    """The verdict for a container weighed on the pad.

    An unknown type (no record) is set aside without looking at its weight.
    """
    if record is None:
        return Verdict(SET_ASIDE, "unknown container type", None)
    net_g = (weighing.mass_kg - record.lightest_empty_mass) * 1000.0
    if not weighing.stable:
        return Verdict(SET_ASIDE, "the weight did not settle", net_g)
    if record.cap_check_required and cap_confirmed is not True:
        return Verdict(SET_ASIDE, "the cap is not confirmed", net_g)
    if net_g < -rule.margin_g:
        return Verdict(SET_ASIDE, "lighter than the lightest empty container of its type", net_g)
    if bin_mask(np.array([net_g]), rule)[0]:
        return Verdict(BIN, "under the limit", net_g)
    return Verdict(SET_ASIDE, "at or over the limit", net_g)
