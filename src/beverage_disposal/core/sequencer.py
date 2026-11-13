"""The sequencer: carries one container through weighing and the decision to bin or set-aside.

It is plain code around the learned skill. The skill picks something up and puts it somewhere;
the sequencer decides where, from the pad (and, for a container the registry does not know, from
the tilt test), under the safety supervisor.
"""

from dataclasses import dataclass
from typing import Protocol

from beverage_disposal.core.coinft import CoinFT
from beverage_disposal.core.config import SceneConfig
from beverage_disposal.core.containers import ContainerSpec
from beverage_disposal.core.fullness import BIN, SET_ASIDE, Verdict, decide
from beverage_disposal.core.fullness_config import FullnessConfig
from beverage_disposal.core.layout import Goal, goals
from beverage_disposal.core.pad import PadSensor
from beverage_disposal.core.registry import Registry
from beverage_disposal.core.supervisor import Event, Outcome, Supervisor
from beverage_disposal.core.tilt import (
    Geometry,
    pad_only_unknown_verdict,
    simulate_measurement,
    tilt_verdict,
)
from beverage_disposal.core.tilt_config import TiltConfig

BINNED = "binned"
SET_ASIDE_AREA = "set_aside"
STOPPED = "stopped"
# The requirement the rule serves: a container of this much or more must never be binned. It is
# fixed here, not read from the rule, so a rule with the wrong threshold still shows as a misbin.
REQUIREMENT_ML = 30.0
UNRECOGNISED = "unrecognised"  # the name an enrolment request gets when perception gives no type


@dataclass(frozen=True)
class Placement:
    """Where a container stands: x, y and the height of the surface it stands on, metres."""

    x: float
    y: float
    surface_z: float


@dataclass(frozen=True)
class Item:
    """A container as presented: what it truly is, and what perception says about it."""

    type_id: str | None  # the registry type perception recognised, or None
    spec: ContainerSpec  # the true size and weight
    volume_ml: float  # the true liquid volume
    density: float  # g/mL
    empty_mass_kg: float  # the true empty weight
    cap_confirmed: bool | None  # what the camera says about a bottle's cap


@dataclass(frozen=True)
class SkillResult:
    success: bool
    reason: str


class Skill(Protocol):
    """The learned skill: pick from `source` (None: where it was spawned) and put at `goal`."""

    def pick_and_place(
        self, source: Placement | None, goal: Goal, *, item: Item, seed: int
    ) -> SkillResult: ...


class IdealSkill:
    """A skill that never fails: for testing the decision on its own."""

    def pick_and_place(
        self, source: Placement | None, goal: Goal, *, item: Item, seed: int
    ) -> SkillResult:
        return SkillResult(True, "ok")


@dataclass(frozen=True)
class Run:
    item: Item
    verdict: Verdict | None
    destination: str  # BINNED, SET_ASIDE_AREA or STOPPED
    events: list[Event]
    threshold_ml: float

    @property
    def misbinned(self) -> bool:
        """Whether a container holding the threshold or more went into the bin."""
        return self.destination == BINNED and self.item.volume_ml >= self.threshold_ml


@dataclass
class Sequencer:
    scene: SceneConfig
    registry: Registry
    fullness: FullnessConfig
    tilt: TiltConfig
    supervisor: Supervisor
    skill: Skill
    pad: PadSensor
    coinft: CoinFT
    use_tilt: bool = True

    def process(self, item: Item, *, seed: int) -> Run:
        mark = len(self.supervisor.events)
        pad_goal = goals(self.scene)["pad"]
        gross = item.empty_mass_kg + item.volume_ml * item.density * 1e-3

        def run(verdict, destination):
            return Run(item, verdict, destination, self.supervisor.events[mark:], REQUIREMENT_ML)

        record, self.registry = (
            self.registry.meet(item.type_id, gross) if item.type_id else (None, self.registry)
        )
        if record is None and item.type_id is None:
            _, self.registry = self.registry.meet(UNRECOGNISED, gross)
        entry = self.supervisor.check_container(record, diameter=item.spec.diameter, mass_kg=gross)
        if entry.outcome is Outcome.SAFE_STOP:
            return run(None, STOPPED)

        first = self.skill.pick_and_place(None, pad_goal, item=item, seed=seed)
        if not first.success:
            self.supervisor.stop("pick_failed", f"the first pick failed: {first.reason}")
            return run(None, STOPPED)

        weighing = self.pad.weigh(gross)
        verdict = self._decide(item, record, weighing)

        destination_goal = goals(self.scene)["bin" if verdict.action == BIN else "set_aside"]
        pad_spot = Placement(pad_goal.center_x, pad_goal.center_y, pad_goal.rest_z)
        second = self.skill.pick_and_place(pad_spot, destination_goal, item=item, seed=seed)
        if not second.success:
            self.supervisor.stop("place_failed", f"the second pick failed: {second.reason}")
            return run(verdict, STOPPED)
        return run(verdict, BINNED if verdict.action == BIN else SET_ASIDE_AREA)

    def _decide(self, item: Item, record, weighing) -> Verdict:
        if record is not None:
            return decide(self.fullness.rule, record, weighing, item.cap_confirmed)
        if not weighing.stable:
            return Verdict(SET_ASIDE, "the weight did not settle", None)
        if not self.use_tilt:
            return pad_only_unknown_verdict(self.fullness.rule, weighing.mass_kg)
        geometry = Geometry(item.spec.diameter / 2, item.spec.height, item.spec.grasp_height)
        torque = simulate_measurement(
            self.tilt, geometry, item.volume_ml, item.empty_mass_kg, item.density, self.coinft
        )
        return tilt_verdict(self.tilt, geometry, weighing.mass_kg, torque)
