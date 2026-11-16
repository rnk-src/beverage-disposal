"""A drawn bottle as the scene's container: its spec for the jaws and the scene built around it.

The bottle is built by hand so every expected number can be worked out here: 0.21 m tall
(0.15 body + 0.04 shoulder + 0.02 neck), a ring 12 mm below the top of the neck.
"""

import dataclasses

import pytest
from support.paths import ROOT

from beverage_disposal.core.bottle_family import (
    GateRecord,
    container_spec_for,
    load_family,
    scene_for_bottle,
)
from beverage_disposal.core.bottles import Bottle, BottleShape
from beverage_disposal.core.config import load_scene_config

SCENE = load_scene_config(ROOT / "configs" / "scene.toml")
SIZE = load_family(ROOT / "configs" / "bottles.toml").classes["pet_500ml"]
SHAPE = BottleShape(
    body_diameter=0.065,
    body_height=0.15,
    shoulder_height=0.04,
    neck_diameter=0.026,
    neck_height=0.02,
    base_style="flat",
    t_side=0.0002,
    t_base=0.0005,
    finish_mass=0.002,
    cap_mass=0.0015,
    label_mass=0.0005,
    support_ring_diameter=0.033,
    support_ring_drop=0.012,
)
BOTTLE = Bottle(SHAPE, density_g_cm3=1.38, size="pet_500ml")
GRIP = {
    "friction": SCENE.container.friction,
    "torsional_friction": SCENE.container.torsional_friction,
}


def test_a_neck_grasp_takes_the_bottle_below_its_ring_at_the_neck_width():
    spec = container_spec_for(BOTTLE, SIZE, **GRIP)
    assert spec.name == "pet_500ml"
    assert spec.diameter == 0.065
    assert spec.height == pytest.approx(0.21)
    assert spec.grasp_diameter == 0.026
    # Top of the neck at 0.21 m, the ring's underside 12 mm below it, the grasp 4 mm under that.
    assert spec.grasp_height == pytest.approx(0.21 - 0.012 - 0.004)
    assert spec.empty_mass == pytest.approx(BOTTLE.empty_mass_kg)
    assert spec.provenance == SIZE.provenance


def test_a_body_grasp_takes_the_bottle_just_below_its_shoulder_at_the_body_width():
    size = dataclasses.replace(SIZE, grasp_mode="body")
    spec = container_spec_for(BOTTLE, size, **GRIP)
    assert spec.grasp_diameter == 0.065
    assert spec.grasp_height == pytest.approx(0.15 - 0.004)


def test_a_neck_grasp_needs_a_ring_to_stay_below():
    no_ring = dataclasses.replace(SHAPE, support_ring_diameter=0.0, support_ring_drop=0.0)
    with pytest.raises(ValueError, match="ring"):
        container_spec_for(dataclasses.replace(BOTTLE, shape=no_ring), SIZE, **GRIP)


def test_the_scene_for_a_bottle_holds_it_with_the_liquid_volume_and_density():
    size = dataclasses.replace(SIZE, grasp_pitch_deg=45.0)
    scene = scene_for_bottle(SCENE, BOTTLE, size, liquid_ml=100.0, density_g_ml=1.05)
    assert scene.bottle is BOTTLE
    assert scene.container == container_spec_for(BOTTLE, size, **GRIP)
    assert scene.fill_ml == 100.0  # the volume: the level and centroid follow it
    assert scene.fill_density_g_ml == 1.05  # the mass follows the density
    assert scene.grasp.pitch_deg == 45.0
    assert (scene.container_x, scene.container_y) == (SCENE.container_x, SCENE.container_y)


def test_making_the_scene_leaves_the_original_scene_alone():
    scene_for_bottle(SCENE, BOTTLE, SIZE, liquid_ml=10.0)
    assert SCENE.bottle is None and SCENE.container.name == "can_12floz"


def test_a_size_that_has_not_passed_the_gate_still_builds_a_scene_for_the_gate_itself():
    untested = dataclasses.replace(SIZE, gate=GateRecord("untested", "2026-11-15", ""))
    assert scene_for_bottle(SCENE, BOTTLE, untested, liquid_ml=0.0).fill_ml == 0.0


def test_the_bottle_takes_the_friction_of_the_container_the_scene_had():
    slippery = dataclasses.replace(SCENE.container, friction=0.7, torsional_friction=0.02)
    scene = scene_for_bottle(
        dataclasses.replace(SCENE, container=slippery), BOTTLE, SIZE, liquid_ml=0.0
    )
    assert (scene.container.friction, scene.container.torsional_friction) == (0.7, 0.02)


def test_a_neck_grasp_opens_the_approach_by_the_rings_overhang():
    # The ring stands (33 - 26) / 2 = 3.5 mm proud of the neck; the jaw must pass it going down.
    scene = scene_for_bottle(SCENE, BOTTLE, SIZE, liquid_ml=0.0)
    assert scene.grasp.clearance == pytest.approx(SCENE.grasp.clearance + 0.0035)


def test_a_body_grasp_keeps_the_scenes_clearance():
    size = dataclasses.replace(SIZE, grasp_mode="body")
    assert scene_for_bottle(SCENE, BOTTLE, size, liquid_ml=0.0).grasp.clearance == (
        SCENE.grasp.clearance
    )
