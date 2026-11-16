"""Monte Carlo of the emptiness policy at decision level: how often a bottle of a volume is binned.

No arm and no physics: the decision depends on the bottle, its weight on the pad and what the
camera reports, not on how the carry goes.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from beverage_disposal.core.bottle_family import BottleFamily, sample_bottle
from beverage_disposal.core.bottles import Bottle
from beverage_disposal.core.fullness import BIN
from beverage_disposal.core.fullness_config import PadSensorConfig
from beverage_disposal.core.materials import MaterialSpec
from beverage_disposal.core.pad import PadSensor, Weighing
from beverage_disposal.core.perception import (
    Perception,
    PerceptionNoise,
    exact_perception,
    perceive,
)
from beverage_disposal.core.policy import EmptinessPolicy
from beverage_disposal.core.statistics import wilson_upper

GRAM = 1e-3  # kilograms per gram
VOLUME_KEY = 10  # volumes are told apart to a tenth of a millilitre in a cell's seed


@dataclass(frozen=True)
class Case:
    """One drawn bottle as it stands on the pad: the truth, the weighing and the camera's look."""

    bottle: Bottle
    volume_ml: float
    weighing: Weighing
    perception: Perception


def draw_case(
    policy: EmptinessPolicy,
    sensor: PadSensor,
    noise: PerceptionNoise | None,
    size: str,
    volume_ml: float,
    rng: np.random.Generator,
    *,
    deformed: bool = False,
    foreign_g: float = 0.0,
) -> Case:
    """Draw a bottle of `size` holding `volume_ml`, with its own drink, wetness, look and weighing.

    The drink's density lies between the rule's lightest and heaviest, the water on the outside
    up to the material's high end. `foreign_g` is a solid object inside: on the scale, invisible
    to the level. Without `noise` the camera is exact.
    """
    rule, material = policy.rule, policy.material
    bottle = sample_bottle(policy.family, size, material, rng)
    density = rng.uniform(rule.min_density_g_per_ml, rule.max_density_g_per_ml)
    wet_g = rng.uniform(material.wet_g.lo, material.wet_g.hi)
    gross = bottle.empty_mass_kg + (volume_ml * density + wet_g + foreign_g) * GRAM
    if noise is None:
        if deformed:
            raise ValueError("a deformed bottle needs the camera noise model to be seen as one")
        seen = exact_perception(bottle, volume_ml)
    else:
        seen = perceive(bottle, volume_ml, noise, rng, deformed=deformed)
    return Case(bottle, volume_ml, sensor.weigh(gross), seen)


@dataclass(frozen=True)
class Cell:
    """One size at one volume: how many bottles were drawn and what became of them."""

    size: str
    volume_ml: float
    trials: int
    binned: int
    upper_95: float  # Wilson 95 % upper bound on the binned rate
    zones: dict[str, int]  # how many fell in each zone; a vetoed bottle has none
    decided_by: dict[str, int]


def misbin_grid(
    policy: EmptinessPolicy,
    family: BottleFamily,
    material: MaterialSpec,
    pad: PadSensorConfig,
    *,
    sizes: tuple[str, ...],
    volumes_ml: tuple[float, ...],
    trials: int,
    seed: int,
    noise: PerceptionNoise | None = None,
) -> list[Cell]:
    """Draw `trials` bottles per size and volume, weigh and judge each, and count the binned.

    Each bottle gets its own drink density (between the rule's lightest and heaviest), its own
    wetness (up to the material's high end) and its own weighing. A cell's draws depend only on
    the seed, the size and the volume. `noise` is the camera's; without it the camera is exact.
    """
    cells = []
    for size in sizes:
        for volume in volumes_ml:
            stream = [seed, list(family.classes).index(size), round(volume * VOLUME_KEY)]
            rng = np.random.default_rng(stream)
            sensor = PadSensor(pad, rng)
            binned = 0
            zones: Counter[str] = Counter()
            decided_by: Counter[str] = Counter()
            for _ in range(trials):
                case = draw_case(policy, sensor, noise, size, volume, rng)
                verdict = policy.decide(case.perception, case.weighing)
                binned += verdict.action == BIN
                decided_by[verdict.decided_by] += 1
                if verdict.zone is not None:
                    zones[verdict.zone] += 1
            cells.append(
                Cell(
                    size,
                    volume,
                    trials,
                    binned,
                    wilson_upper(binned, trials),
                    dict(zones),
                    dict(decided_by),
                )
            )
    return cells
