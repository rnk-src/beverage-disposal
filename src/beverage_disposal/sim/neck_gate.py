"""Runs the grasp gate: seeded scripted carries of a bottle from the pad, measured for the bar."""

import math

import numpy as np

from beverage_disposal.core.bottle_family import (
    BottleFamily,
    SizeClass,
    container_spec_for,
    sample_bottle,
    scene_for_bottle,
)
from beverage_disposal.core.bottles import Bottle, capacity_ml
from beverage_disposal.core.config import SceneConfig
from beverage_disposal.core.gate import GateBar, GateRow, grip_slip_m, swing_settle_s
from beverage_disposal.core.layout import goals
from beverage_disposal.core.materials import MaterialSpec
from beverage_disposal.core.reach import container_poses, steepest_common_pitch
from beverage_disposal.core.scripted import Phase, ScriptedController
from beverage_disposal.core.sequencer import INTAKE_SCATTER
from beverage_disposal.sim.rollout import run_episode

FILLS = ("empty", "payload")
WATER_KG_PER_ML = 1e-3


def gate_fill_ml(bottle: Bottle, fill: str, payload_kg: float) -> float:
    """Millilitres of water for a gate fill: none, or up to the payload limit or the brim."""
    if fill == "empty":
        return 0.0
    if fill != "payload":
        raise ValueError(f"fill must be one of {FILLS}, got {fill!r}")
    room_ml = (payload_kg - bottle.empty_mass_kg) / WATER_KG_PER_ML
    return min(capacity_ml(bottle.shape), room_ml)


def class_pitch_deg(
    scene: SceneConfig,
    family: BottleFamily,
    material: MaterialSpec,
    size: SizeClass,
    draws: int = 200,
) -> int | None:
    """The steepest tool pitch that reaches every pose for the size's tallest and shortest bottle.

    None when some pose of the leg is out of reach at every pitch.
    """
    can = scene.container
    specs = [
        container_spec_for(
            sample_bottle(family, size.name, material, np.random.default_rng(seed)),
            size,
            friction=can.friction,
            torsional_friction=can.torsional_friction,
        )
        for seed in range(draws)
    ]
    ends = (max(specs, key=lambda spec: spec.height), min(specs, key=lambda spec: spec.height))
    return steepest_common_pitch([pose for spec in ends for pose in container_poses(scene, spec)])


def _step_of(log: list[tuple[Phase, int]], phase: Phase) -> int:
    """The control step `phase` began on, or -1 when the run never reached it."""
    return next((step for logged, step in log if logged is phase), -1)


def run_gate(
    scene: SceneConfig,
    mjcf,
    family: BottleFamily,
    material: MaterialSpec,
    size: SizeClass,
    fill: str,
    bar: GateBar,
    *,
    payload_kg: float,
    seeds: range | None = None,
    squeeze: bool = True,
) -> GateRow:
    """Carry a freshly drawn bottle of `size` from the pad to the set-aside area, once per seed.

    The bottle, where it stands on the pad and the run all follow the seed. `squeeze=False` is
    the broken grasp the gate must turn down.
    """
    slips, settles, places = [], [], []
    for seed in range(bar.seeds) if seeds is None else seeds:
        bottle = sample_bottle(family, size.name, material, np.random.default_rng(seed))
        config = scene_for_bottle(scene, bottle, size, gate_fill_ml(bottle, fill, payload_kg))
        jitter = np.random.default_rng([seed, 2]).uniform(-INTAKE_SCATTER, INTAKE_SCATTER, 2)
        pad = scene.pad
        start = (pad.center_x + jitter[0], pad.center_y + jitter[1], pad.height)
        controller = ScriptedController(config, squeeze=squeeze)
        goal = goals(config)["set_aside"]
        episode = run_episode(config, mjcf, controller, goal, seed=seed, start=start)
        log = controller.phase_log
        grip, carry_end = _step_of(log, Phase.LIFT), _step_of(log, Phase.LOWER)
        release = _step_of(log, Phase.RELEASE)
        assert episode.trajectory is not None
        slips.append(grip_slip_m(episode.trajectory, grip, release))
        settles.append(
            swing_settle_s(episode.trajectory, carry_end, bar.swing_band_rad, release)
            if release > 0
            else math.inf
        )
        places.append(episode.success)
    return GateRow(size.name, fill, tuple(slips), tuple(settles), tuple(places))
