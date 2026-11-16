"""The emptiness policy for PET bottles: hard vetoes, then the physics zones.

Rule-only for now: a bottle is binned only when its weight proves it holds under the threshold.
The classifier that may overturn the ambiguous zone comes later; without it that zone is set aside.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from beverage_disposal.core.bottle_family import BottleFamily, classes_matching
from beverage_disposal.core.bounds import empty_mass_bounds
from beverage_disposal.core.config import from_mapping
from beverage_disposal.core.fullness import BIN, SET_ASIDE, Verdict
from beverage_disposal.core.fullness_config import RuleConfig
from beverage_disposal.core.materials import MaterialSpec
from beverage_disposal.core.pad import Weighing
from beverage_disposal.core.perception import Perception
from beverage_disposal.core.zones import Zone, liquid_bounds_ml, zone_of

PET = "PET"


@dataclass(frozen=True)
class PolicyConfig:
    alpha: float
    delta: float
    volume_bins: tuple[float, ...]
    veto_score: float
    min_confidence: float
    allow_overturn: bool


def load_policy_config(path: str | Path) -> PolicyConfig:
    with open(path, "rb") as file:
        return from_mapping(PolicyConfig, tomllib.load(file), "policy")


@dataclass(frozen=True)
class EmptinessPolicy:
    rule: RuleConfig
    material: MaterialSpec
    family: BottleFamily
    config: PolicyConfig
    # The sizes the policy acts on; None means the sizes whose grasp gate passed.
    sizes: tuple[str, ...] | None = None

    def supported(self) -> tuple[str, ...]:
        return self.family.supported() if self.sizes is None else self.sizes

    def decide(self, perception: Perception, weighing: Weighing) -> Verdict:
        """Bin or set aside a bottle standing on the pad. The vetoes come first, in this order."""
        if perception.material != PET:
            return _veto("not_pet", f"the material is {perception.material}, not PET")
        if not weighing.stable:
            return _veto("unsettled", "the weight did not settle")
        if perception.confidence < self.config.min_confidence:
            return _veto("low_confidence", f"perception is only {perception.confidence:.2f} sure")
        matches = classes_matching(self.family, perception.dims)
        if not matches or not set(matches) <= set(self.supported()):
            return _veto("unsupported_size", f"the size matches {matches or 'no class'}")

        bounds = empty_mass_bounds(perception.dims, self.material)
        v_lo, v_hi = liquid_bounds_ml(weighing.mass_kg, bounds, self.rule)
        if v_hi < 0:
            return _veto("lighter_than_possible", "lighter than the lightest bottle of its size")
        zone = zone_of(v_lo, v_hi, self.rule.threshold_ml)
        action, reason = {
            Zone.SURELY_UNDER: (BIN, "the weight proves it holds under the threshold"),
            Zone.SURELY_OVER: (SET_ASIDE, "the weight proves it holds the threshold or more"),
            Zone.AMBIGUOUS: (SET_ASIDE, "the weight cannot tell"),
        }[zone]
        return Verdict(action, reason, None, zone=zone.value, bounds_ml=(v_lo, v_hi))


def _veto(code: str, reason: str) -> Verdict:
    return Verdict(SET_ASIDE, reason, None, decided_by=f"veto:{code}")
