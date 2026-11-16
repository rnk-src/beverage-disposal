"""The sequencer: weighs and judges a container on the pad, then carries it once.

A person puts the container on the weighing pad. The sequencer weighs it there, lets the safety
supervisor refuse it before any lift, decides bin or set-aside, and calls the skill once to carry
it. It is plain code around the learned skill and never leaves the decision to it. A container
the registry does not know is always set aside.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from beverage_disposal.core.bottles import Bottle
from beverage_disposal.core.config import SceneConfig
from beverage_disposal.core.containers import ContainerSpec
from beverage_disposal.core.fullness import BIN, Verdict, decide
from beverage_disposal.core.fullness_config import FullnessConfig
from beverage_disposal.core.layout import Goal, goals
from beverage_disposal.core.pad import PadSensor, Weighing
from beverage_disposal.core.perception import (
    Perception,
    PerceptionNoise,
    exact_perception,
    perceive,
)
from beverage_disposal.core.policy import PET, EmptinessPolicy
from beverage_disposal.core.registry import Registry
from beverage_disposal.core.supervisor import Event, Outcome, Supervisor

BINNED = "binned"
SET_ASIDE_AREA = "set_aside"
STOPPED = "stopped"
# The requirement the rule serves: a container of this much or more must never be binned. It is
# fixed here, not read from the rule, so a rule with the wrong threshold still shows as a misbin.
REQUIREMENT_ML = 30.0
# A person puts the container on the pad this far off its centre at most, each way (assumed).
INTAKE_SCATTER = 0.005
PERCEPTION_STREAM = 4  # the seed stream of the camera's errors
INTAKE_STREAM = 3  # the seed stream of the placement scatter, apart from the pad noise
GRAM = 1e-3  # kilograms per gram
VETO = "veto:"  # how a verdict names a veto in `decided_by`
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
    bottle: Bottle | None = None  # the true shape, when it is a PET bottle
    material: str = "aluminium"
    wet_g: float = 0.0  # water on the outside: on the scale, not in the simulated body


@dataclass(frozen=True)
class SkillResult:
    success: bool
    reason: str


class Skill(Protocol):
    """The learned skill: pick from `source` (None: a seeded spawn) and put at `goal`."""

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
    supervisor: Supervisor
    skill: Skill
    pad: PadSensor
    policy: EmptinessPolicy | None = None  # judges PET bottles; without it they are set aside
    perception_noise: PerceptionNoise | None = None  # the camera's errors; None is an exact camera

    def _perceive(self, item: Item, seed: int) -> Perception | None:
        """What the camera reports for a bottle on the pad; nothing for other containers."""
        if item.bottle is None:
            return None
        if self.perception_noise is None:
            return exact_perception(item.bottle, item.volume_ml, item.material)
        rng = np.random.default_rng([seed, PERCEPTION_STREAM])
        return perceive(
            item.bottle, item.volume_ml, self.perception_noise, rng, material=item.material
        )

    @staticmethod
    def _grasp_width(item: Item, perception: Perception | None) -> tuple[float, float]:
        """The interval for the width the jaws close on: the camera's for a bottle, else exact."""
        width = item.spec.grasp_diameter
        if perception is None or item.bottle is None:
            return (width, width)
        by_the_neck = width == item.bottle.shape.neck_diameter
        return perception.dims.neck_diameter if by_the_neck else perception.dims.body_diameter

    def _decide(self, item: Item, perception: Perception | None, weighing: Weighing) -> Verdict:
        """The verdict for a container on the pad; a set-aside a person should see is reported."""
        if self.policy is not None and perception is not None and item.material == PET:
            verdict = self.policy.decide(perception, weighing)
            if verdict.decided_by.startswith(VETO):
                self.supervisor.set_aside(verdict.decided_by.removeprefix(VETO), verdict.reason)
            return verdict
        record, self.registry = self.registry.meet(item.type_id or UNRECOGNISED, weighing.mass_kg)
        if record is None:
            self.supervisor.set_aside("unknown_type", "unknown container type: set aside")
        return decide(self.fullness.rule, record, weighing, item.cap_confirmed)

    def process(self, item: Item, *, seed: int) -> Run:
        mark = len(self.supervisor.events)
        targets = goals(self.scene)
        gross = item.empty_mass_kg + (item.volume_ml * item.density + item.wet_g) * GRAM

        def run(verdict, destination):
            return Run(item, verdict, destination, self.supervisor.events[mark:], REQUIREMENT_ML)

        weighing = self.pad.weigh(gross)
        perception = self._perceive(item, seed)
        entry = self.supervisor.check_container(
            grasp_width=self._grasp_width(item, perception), weighing=weighing
        )
        if entry.outcome is Outcome.SAFE_STOP:
            return run(None, STOPPED)

        verdict = self._decide(item, perception, weighing)

        binned = verdict.action == BIN
        pad = self.scene.pad
        off = np.random.default_rng([seed, INTAKE_STREAM]).uniform(
            -INTAKE_SCATTER, INTAKE_SCATTER, 2
        )
        spot = Placement(pad.center_x + float(off[0]), pad.center_y + float(off[1]), pad.height)
        goal = targets["bin" if binned else "set_aside"]
        carry = self.skill.pick_and_place(spot, goal, item=item, seed=seed)
        if not carry.success:
            self.supervisor.stop("place_failed", f"the carry failed: {carry.reason}")
            return run(verdict, STOPPED)
        return run(verdict, BINNED if binned else SET_ASIDE_AREA)
