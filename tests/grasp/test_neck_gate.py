"""The gate run in the real simulation: it measures a carry, and a grasp that cannot hold fails."""

import math

import numpy as np
import pytest
from support.paths import ROOT
from support.so101 import MJCF, SO101

from beverage_disposal.core.bottle_family import container_spec_for, load_family, sample_bottle
from beverage_disposal.core.bottles import capacity_ml
from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.gate import load_gate_bar, passes
from beverage_disposal.core.ik import solve_ik
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.reach import container_poses
from beverage_disposal.sim.neck_gate import class_pitch_deg, gate_fill_ml, run_gate

SCENE = load_scene_config(ROOT / "configs" / "scene.toml")
FAMILY = load_family(ROOT / "configs" / "bottles.toml")
PET = load_materials(ROOT / "configs" / "materials.toml")["PET"]
BAR = load_gate_bar(ROOT / "configs" / "gate.toml")
SMALL = FAMILY.classes["pet_250ml"]
BOTTLE = sample_bottle(FAMILY, "pet_250ml", PET, np.random.default_rng(0))


def test_the_empty_fill_is_no_liquid_and_an_unknown_fill_is_refused():
    assert gate_fill_ml(BOTTLE, "empty", 0.6) == 0.0
    with pytest.raises(ValueError, match="fill"):
        gate_fill_ml(BOTTLE, "half", 0.6)


def test_the_payload_fill_brings_the_bottle_to_the_payload_or_to_its_brim():
    # A small bottle cannot hold 0.6 kg: it is filled to the brim.
    assert gate_fill_ml(BOTTLE, "payload", 0.6) == pytest.approx(capacity_ml(BOTTLE.shape))
    # With a 0.1 kg payload the water is what is left after the empty bottle, at 1 g per mL.
    assert gate_fill_ml(BOTTLE, "payload", 0.1) == pytest.approx(
        (0.1 - BOTTLE.empty_mass_kg) * 1000
    )


@pytest.mark.medium
def test_a_gate_run_gives_one_measurement_per_seed():
    row = run_gate(
        SCENE, SO101 / MJCF, FAMILY, PET, SMALL, "empty", BAR, payload_kg=0.6, seeds=range(2)
    )
    assert (row.size, row.fill) == ("pet_250ml", "empty")
    assert len(row.slips) == len(row.settle_times) == len(row.places_upright) == 2
    # Both runs reach the carry, so the slip is a real length.
    assert all(math.isfinite(slip) and slip >= 0 for slip in row.slips)


@pytest.mark.medium
def test_a_grasp_that_never_squeezes_does_not_pass_the_gate():
    row = run_gate(
        SCENE,
        SO101 / MJCF,
        FAMILY,
        PET,
        SMALL,
        "empty",
        BAR,
        payload_kg=0.6,
        seeds=range(2),
        squeeze=False,
    )
    assert not any(row.places_upright)
    assert not passes(row, BAR)


def test_the_class_pitch_reaches_every_pose_of_its_tallest_bottle_and_5_degrees_more_does_not():
    size = FAMILY.classes["pet_500ml"]
    pitch = class_pitch_deg(SCENE, FAMILY, PET, size, draws=40)
    can = SCENE.container
    specs = [
        container_spec_for(
            sample_bottle(FAMILY, "pet_500ml", PET, np.random.default_rng(seed)),
            size,
            friction=can.friction,
            torsional_friction=can.torsional_friction,
        )
        for seed in range(40)
    ]
    tallest = max(specs, key=lambda spec: spec.height)
    poses = [position for _, position in container_poses(SCENE, tallest)]

    def all_reached(degrees):
        return all(solve_ik(np.array(p), math.radians(degrees)).converged for p in poses)

    assert pitch is not None and all_reached(pitch) and not all_reached(pitch + 5)


def test_a_size_with_a_pose_out_of_reach_has_no_class_pitch():
    assert class_pitch_deg(SCENE, FAMILY, PET, FAMILY.classes["pet_2000ml"], draws=40) is None
