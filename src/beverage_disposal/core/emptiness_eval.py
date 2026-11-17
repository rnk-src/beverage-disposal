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
from beverage_disposal.core.probes import NO_PROBES, ProbeModels, ProbeReadings, probe
from beverage_disposal.core.statistics import wilson_upper

CLASSIFIER = "classifier"
GRAM = 1e-3  # kilograms per gram
VOLUME_KEY = 10  # volumes are told apart to a tenth of a millilitre in a cell's seed
PROBE_STREAM = 7  # the probes' own generator, apart from the bottle's
KG_TO_ML_OF_WATER = 1000.0


@dataclass(frozen=True)
class Case:
    """One drawn bottle as it stands on the pad: the truth, the weighing and the camera's look."""

    bottle: Bottle
    volume_ml: float
    weighing: Weighing
    perception: Perception
    probes: ProbeReadings = NO_PROBES


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
    probes: ProbeModels | None = None,
    probe_rng: np.random.Generator | None = None,
    liquid_gain: float = 1.0,
    density_g_per_ml: tuple[float, float] | None = None,
) -> Case:
    """Draw a bottle of `size` holding `volume_ml`, with its own drink, wetness, look and weighing.

    The drink's density lies between the rule's lightest and heaviest, the water on the outside
    up to the material's high end. `foreign_g` is a solid object inside: on the scale, invisible
    to the level. Without `noise` the camera is exact. Fitted `probes` read from `probe_rng`, never
    from `rng`, so fitting them leaves every bottle as it was. `density_g_per_ml` replaces the
    rule's drink range, for a thick drink (slushie, shake) heavier than the rule assumes.
    """
    rule, material = policy.rule, policy.material
    bottle = sample_bottle(policy.family, size, material, rng)
    low, high = density_g_per_ml or (rule.min_density_g_per_ml, rule.max_density_g_per_ml)
    density = rng.uniform(low, high)
    wet_g = rng.uniform(material.wet_g.lo, material.wet_g.hi)
    gross = bottle.empty_mass_kg + (volume_ml * density + wet_g + foreign_g) * GRAM
    if noise is None:
        if deformed:
            raise ValueError("a deformed bottle needs the camera noise model to be seen as one")
        seen = exact_perception(bottle, volume_ml)
    else:
        seen = perceive(bottle, volume_ml, noise, rng, deformed=deformed)
    weighing = sensor.weigh(gross)
    if probes is None:
        readings = NO_PROBES
    elif probe_rng is None:
        raise ValueError("probes need a probe_rng of their own")
    else:
        readings = probe(bottle, volume_ml, probes, probe_rng, liquid_gain=liquid_gain)
    return Case(bottle, volume_ml, weighing, seen, readings)


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
    overturned: int = 0  # binned by the classifier in the ambiguous zone
    deformed: bool = False  # the bottles were crushed
    foreign_g: float = 0.0  # a solid object of this mass was inside each bottle
    probes: str = ""  # which probes were fitted: "", "wall", "liquid" or "wall+liquid"
    liquid_gain: float = 1.0  # under 1: the liquid probe saw less than was there
    density_g_per_ml: tuple[float, float] | None = None  # a thick drink's density range


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
    stream: tuple[int, ...] = (),
    deformed: bool = False,
    foreign_g: float = 0.0,
    probes: ProbeModels | None = None,
    liquid_gain: float = 1.0,
    density_g_per_ml: tuple[float, float] | None = None,
) -> list[Cell]:
    """Draw `trials` bottles per size and volume, weigh and judge each, and count the binned.

    Each bottle gets its own drink density (between the rule's lightest and heaviest), its own
    wetness (up to the material's high end) and its own weighing. A cell's draws depend only on
    the seed, the stream, the size and the volume. `noise` is the camera's; without it the camera
    is exact. `stream` keeps a grid's bottles apart from another's on the same seed. `deformed`
    crushes every bottle and `foreign_g` puts a solid object of that mass in each. Fitted `probes`
    draw from a generator of their own per cell; `liquid_gain` is what the liquid probe sees of
    the liquid (see `probe`). `density_g_per_ml` draws every drink from that range instead.
    """
    fitted = ""
    if probes is not None:
        fitted = "+".join(
            name
            for name, model in (("wall", probes.wall), ("liquid", probes.liquid))
            if model is not None
        )
    cells = []
    for size in sizes:
        for volume in volumes_ml:
            index = list(family.classes).index(size)
            rng = np.random.default_rng([seed, *stream, index, round(volume * VOLUME_KEY)])
            probe_rng = np.random.default_rng(
                [seed, *stream, index, round(volume * VOLUME_KEY), PROBE_STREAM]
            )
            sensor = PadSensor(pad, rng)
            binned = overturned = 0
            zones: Counter[str] = Counter()
            decided_by: Counter[str] = Counter()
            for _ in range(trials):
                case = draw_case(
                    policy,
                    sensor,
                    noise,
                    size,
                    volume,
                    rng,
                    deformed=deformed,
                    foreign_g=foreign_g,
                    probes=probes,
                    probe_rng=probe_rng,
                    liquid_gain=liquid_gain,
                    density_g_per_ml=density_g_per_ml,
                )
                verdict = policy.decide(case.perception, case.weighing, case.probes)
                binned += verdict.action == BIN
                overturned += verdict.action == BIN and verdict.decided_by == CLASSIFIER
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
                    overturned,
                    deformed,
                    foreign_g,
                    fitted,
                    liquid_gain,
                    density_g_per_ml,
                )
            )
    return cells


def payload_volume_ml(family: BottleFamily, size: str, *, payload_kg: float) -> float:
    """The most liquid a bottle of `size` is tested with, mL: its size, or the payload as water."""
    return min(family.classes[size].nominal_ml, payload_kg * KG_TO_ML_OF_WATER)


def summarize_cells(cells: list[Cell], *, threshold_ml: float) -> dict:
    """The headline counts of a grid. Crushed and foreign-object cells are counted apart."""
    sound = [c for c in cells if _is_sound(c)]
    over = [c for c in sound if c.volume_ml >= threshold_ml]
    vetoes: Counter[str] = Counter()
    for cell in cells:
        vetoes.update({k: n for k, n in cell.decided_by.items() if k.startswith("veto:")})
    foreign: dict[str, dict[str, list[int]]] = {}
    for cell in cells:
        if cell.foreign_g:
            foreign.setdefault(cell.size, {})[f"{cell.foreign_g:g}"] = [cell.binned, cell.trials]
    under_reading: dict[str, dict[str, dict[str, list[int]]]] = {}
    for cell in cells:
        if cell.liquid_gain != 1.0 and cell.density_g_per_ml is None:
            by_gain = under_reading.setdefault(cell.size, {}).setdefault(
                f"{cell.liquid_gain:g}", {}
            )
            by_gain[f"{cell.volume_ml:g}"] = [cell.binned, cell.trials]
    thick: dict[str, dict[str, dict[str, list[int]]]] = {}
    for cell in cells:
        if cell.density_g_per_ml is not None:
            by_gain = thick.setdefault(cell.size, {}).setdefault(f"{cell.liquid_gain:g}", {})
            by_gain[f"{cell.volume_ml:g}"] = [cell.binned, cell.trials]
    return {
        "binned_at_or_over_threshold": sum(c.binned for c in over),
        "trials_at_or_over_threshold": sum(c.trials for c in over),
        "worst_upper_95_at_or_over_threshold": max((c.upper_95 for c in over), default=None),
        "empties_binned": {c.size: [c.binned, c.trials] for c in sound if c.volume_ml == 0.0},
        "overturned": sum(c.overturned for c in cells),
        "vetoes": dict(vetoes),
        "crushed_binned": {c.size: [c.binned, c.trials] for c in cells if c.deformed},
        "foreign_binned": foreign,
        "under_reading_binned": under_reading,
        "thick_drink_binned": thick,
    }


def _is_sound(cell: Cell) -> bool:
    """Not crushed, no solid, no thick drink and no probe that sees less than is there."""
    return (
        not cell.deformed
        and cell.foreign_g == 0.0
        and cell.liquid_gain == 1.0
        and cell.density_g_per_ml is None
    )


def target_met(
    cells: list[Cell], *, volumes_ml: tuple[float, ...], max_set_aside: float
) -> dict[str, bool]:
    """Per size: whether at most `max_set_aside` of each volume's sound bottles were set aside."""
    found = {(c.size, c.volume_ml): c for c in cells if _is_sound(c)}
    result: dict[str, bool] = {}
    for size in dict.fromkeys(c.size for c in cells):
        met = True
        for volume in volumes_ml:
            if (size, volume) not in found:
                raise ValueError(f"no sound cell of {size} at {volume:g} mL")
            cell = found[(size, volume)]
            met = met and cell.binned / cell.trials >= 1 - max_set_aside
        result[size] = met
    return result
