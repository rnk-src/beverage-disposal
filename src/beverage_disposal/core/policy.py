"""The emptiness policy for PET bottles: hard vetoes, then the physics zones, then the classifier.

Without a bundle the policy is rule-only: a bottle is binned only when its weight proves it holds
under the threshold. With one, the classifier may veto that bin, and may bin a bottle of the
ambiguous zone whose score beats its size's calibrated threshold. It never touches a bottle whose
weight proves the threshold or more.
"""

import dataclasses
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from beverage_disposal.core.bottle_family import BottleFamily, classes_matching
from beverage_disposal.core.bounds import EmptyMassBounds, empty_mass_bounds
from beverage_disposal.core.classifier import ModelBundle
from beverage_disposal.core.config import from_mapping
from beverage_disposal.core.features import feature_vector, level_volume_bounds_ml
from beverage_disposal.core.fullness import BIN, SET_ASIDE, Verdict
from beverage_disposal.core.fullness_config import RuleConfig
from beverage_disposal.core.materials import MaterialSpec, with_side_wall
from beverage_disposal.core.pad import Weighing
from beverage_disposal.core.perception import Perception
from beverage_disposal.core.probes import NO_PROBES, ProbeReadings
from beverage_disposal.core.zones import Zone, intersect_ml, liquid_bounds_ml, zone_of

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
    # The liquid probe's reading is used only when the weight alone proves at most this much liquid:
    # contents the probe cannot see (ice, a solid) then weigh at most about that much.
    probe_reach_ml: float = math.inf


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
    bundle: ModelBundle | None = None  # the classifier; None is the rule-only policy

    def supported(self) -> tuple[str, ...]:
        return self.family.supported() if self.sizes is None else self.sizes

    def assess(
        self, perception: Perception, weighing: Weighing, probes: ProbeReadings = NO_PROBES
    ) -> "Assessment":
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

        wall = probes.side_wall_mm
        material = self.material if wall is None else with_side_wall(self.material, wall)
        if material is None:
            return _vetoed("implausible_wall", "the measured wall is outside the material's range")

        bounds = empty_mass_bounds(perception.dims, material)
        v_bounds = liquid_bounds_ml(weighing.mass_kg, bounds, self.rule)
        # The level's volume uses the material as it was, so a wall reading does not move it.
        level_ml = level_volume_bounds_ml(perception, self.material)
        found = Assessment(None, matches, bounds, v_bounds, None, level_ml, v_bounds)
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

        reading = probes.liquid_ml
        if reading is not None and level_ml is not None:
            if level_ml[0] > reading[1] + self.config.level_slack_ml:
                # The camera's level is used only to set a bottle aside, never to bin one.
                return dataclasses.replace(
                    found,
                    veto=_veto(
                        "inconsistent_level_and_probe",
                        "the camera sees more inside than the liquid probe does",
                    ),
                )
        if reading is not None and v_bounds[1] > self.config.probe_reach_ml:
            reading = None  # the weight leaves too much room for contents the probe cannot see
        fused = v_bounds if reading is None else intersect_ml(v_bounds, reading)
        if fused is None:
            return dataclasses.replace(
                found,
                veto=_veto(
                    "inconsistent_weight_and_probe", "the weight and the liquid probe disagree"
                ),
            )
        return dataclasses.replace(
            found, v_bounds=fused, zone=zone_of(*fused, self.rule.threshold_ml)
        )

    def decide(
        self, perception: Perception, weighing: Weighing, probes: ProbeReadings = NO_PROBES
    ) -> Verdict:
        """Bin or set aside a bottle standing on the pad."""
        found = self.assess(perception, weighing, probes)
        if found.veto is not None:
            return found.veto
        assert found.zone is not None and found.bounds is not None and found.v_bounds is not None
        zone = found.zone

        def verdict(action: str, reason: str, by: str = "rule", score: float | None = None):
            return Verdict(
                action,
                reason,
                None,
                zone.value,
                decided_by=by,
                score=score,
                bounds_ml=found.v_bounds,
            )

        if zone is Zone.SURELY_OVER:
            return verdict(SET_ASIDE, "the weight proves it holds the threshold or more")
        if self.bundle is None:
            if zone is Zone.SURELY_UNDER:
                return verdict(BIN, "the weight proves it holds under the threshold")
            return verdict(SET_ASIDE, "the weight cannot tell")

        x = feature_vector(
            perception, weighing, found.bounds, found.v_bounds, self.family, self.material
        )
        if not self.bundle.ranges.in_distribution(x):
            return _veto("out_of_distribution", "unlike any bottle the classifier was trained on")
        try:
            score = float(self.bundle.model.score(x))
        except Exception as error:  # a broken model must never bin, whatever it raises
            return _veto("model_error", f"the classifier failed: {error}")
        if math.isnan(score):
            return _veto("model_error", "the classifier gave no score")

        if zone is Zone.SURELY_UNDER:
            if score < self.config.veto_score:
                return verdict(SET_ASIDE, "the classifier doubts it is empty", "classifier", score)
            return verdict(BIN, "the weight proves it holds under the threshold", score=score)
        # Ambiguous: the bottle may be of any size it matches, so its score must beat the
        # strictest of their thresholds. A size with no threshold has an infinite one.
        tau = max(self.bundle.tau_by_size.get(name, math.inf) for name in found.matches)
        if self.config.allow_overturn and score > tau:
            return verdict(BIN, "the classifier is sure it is empty", "classifier", score)
        return verdict(SET_ASIDE, "the weight cannot tell", score=score)


@dataclass(frozen=True)
class Assessment:
    """What the policy worked out about one bottle before deciding."""

    veto: Verdict | None
    matches: tuple[str, ...] = ()
    bounds: EmptyMassBounds | None = None
    v_bounds: tuple[float, float] | None = None  # least and most liquid the evidence allows, mL
    zone: Zone | None = None  # None when vetoed
    level_ml: tuple[float, float] | None = None  # the liquid the seen level stands for
    weight_ml: tuple[float, float] | None = None  # what the weight alone allows, mL


def _vetoed(code: str, reason: str) -> Assessment:
    return Assessment(_veto(code, reason))


def _veto(code: str, reason: str) -> Verdict:
    return Verdict(SET_ASIDE, reason, None, decided_by=f"veto:{code}")
