"""The emptiness policy for PET bottles: hard vetoes, then the physics zones.

Rule-only for now: a bottle is binned only when its weight proves it holds under the threshold.
The classifier that may overturn the ambiguous zone comes later; without it that zone is set aside.
"""

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path

from beverage_disposal.core.bottle_family import BottleFamily, classes_matching
from beverage_disposal.core.bounds import EmptyMassBounds, empty_mass_bounds
from beverage_disposal.core.config import from_mapping
from beverage_disposal.core.features import level_volume_bounds_ml
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
    level_slack_ml: float  # the weight may show this much more liquid than the level before a veto


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

    def assess(self, perception: Perception, weighing: Weighing) -> "Assessment":
        """The vetoes, in order, and then what the weight says. Nothing is decided here."""
        if perception.material != PET:
            return _vetoed("not_pet", f"the material is {perception.material}, not PET")
        if not weighing.stable:
            return _vetoed("unsettled", "the weight did not settle")
        if perception.confidence < self.config.min_confidence:
            return _vetoed("low_confidence", f"perception is only {perception.confidence:.2f} sure")
        matches = classes_matching(self.family, perception.dims)
        if not matches or not set(matches) <= set(self.supported()):
            return _vetoed("unsupported_size", f"the size matches {matches or 'no class'}")

        bounds = empty_mass_bounds(perception.dims, self.material)
        v_bounds = liquid_bounds_ml(weighing.mass_kg, bounds, self.rule)
        level_ml = level_volume_bounds_ml(perception, self.material)
        found = Assessment(None, matches, bounds, v_bounds, None, level_ml)
        if v_bounds[1] < 0:
            return dataclasses.replace(
                found, veto=_veto("lighter_than_possible", "lighter than any bottle of its size")
            )
        if level_ml is not None and level_ml[1] + self.config.level_slack_ml < v_bounds[0]:
            # More weight than the visible liquid explains: something else is in the bottle.
            return dataclasses.replace(
                found,
                veto=_veto(
                    "inconsistent_weight_and_level",
                    "the weight shows more inside than the level does",
                ),
            )
        return dataclasses.replace(found, zone=zone_of(*v_bounds, self.rule.threshold_ml))

    def decide(self, perception: Perception, weighing: Weighing) -> Verdict:
        """Bin or set aside a bottle standing on the pad."""
        found = self.assess(perception, weighing)
        if found.veto is not None:
            return found.veto
        assert found.zone is not None
        action, reason = {
            Zone.SURELY_UNDER: (BIN, "the weight proves it holds under the threshold"),
            Zone.SURELY_OVER: (SET_ASIDE, "the weight proves it holds the threshold or more"),
            Zone.AMBIGUOUS: (SET_ASIDE, "the weight cannot tell"),
        }[found.zone]
        return Verdict(action, reason, None, zone=found.zone.value, bounds_ml=found.v_bounds)


@dataclass(frozen=True)
class Assessment:
    """What the policy worked out about one bottle before deciding."""

    veto: Verdict | None
    matches: tuple[str, ...] = ()
    bounds: EmptyMassBounds | None = None
    v_bounds: tuple[float, float] | None = None  # least and most liquid the weight allows, mL
    zone: Zone | None = None  # None when vetoed
    level_ml: tuple[float, float] | None = None  # the liquid the seen level stands for


def _vetoed(code: str, reason: str) -> Assessment:
    return Assessment(_veto(code, reason))


def _veto(code: str, reason: str) -> Verdict:
    return Verdict(SET_ASIDE, reason, None, decided_by=f"veto:{code}")
