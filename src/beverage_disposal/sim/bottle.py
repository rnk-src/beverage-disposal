"""A PET bottle as MuJoCo geoms, standing on its base with the liquid welded inside.

The body, shoulder, neck and ring are solid collision shapes that carry the shell's mass; a real
shell is hollow, so their moments of inertia are a little too large. That slows a swing slightly
and is accepted. The liquid is a point mass at its closed-form centroid (the rigid model, as
locked) with the inertia of a solid cylinder; a separate, non-colliding geom shows its level.
"""

import math

import mujoco
import numpy as np

from beverage_disposal.core.bottles import (
    G_CM3_TO_KG_M3,
    Bottle,
    areas,
    level_for_volume_m,
    liquid_centroid_m,
    total_height_m,
)

SEGMENTS = 32  # sides of the shoulder's cone
RING_THICKNESS = 0.002  # metres
LEVEL_RGBA = [0.2, 0.4, 0.9, 0.5]
LEVEL_INSET = 0.95  # the drawn liquid is a little narrower than the wall so it does not z-fight
VISUAL_GROUP = 2
LIQUID_KG_PER_ML = 1e-3  # the scene counts its fill as water


def add_bottle(
    spec: mujoco.MjSpec,
    world,
    bottle: Bottle,
    liquid_ml: float,
    *,
    x: float,
    y: float,
    friction: list[float],
    condim: int,
) -> None:
    """Add the bottle as a free body named "container", standing on the table at (x, y)."""
    shape = bottle.shape
    total = total_height_m(shape)
    half_total = total / 2
    found = areas(shape)
    plastic = bottle.density_g_cm3 * G_CM3_TO_KG_M3
    body_mass = plastic * (found.side * shape.t_side + found.base * shape.t_base)
    shoulder_mass = plastic * found.shoulder * shape.t_side
    # The ring is part of the finish: it takes half of the finish's mass, the neck the other half.
    ring_mass = shape.finish_mass / 2
    neck_mass = shape.finish_mass - ring_mass + shape.cap_mass + shape.label_mass

    # Heights below are above the base; the body's origin is at half the total height.
    def local(z: float) -> float:
        return z - half_total

    container = world.add_body(name="container", pos=[x, y, half_total])
    container.add_freejoint(name="container")

    def collider(name, kind, size, z, mass, **extra):
        return container.add_geom(
            name=name,
            type=kind,
            size=size,
            pos=[0.0, 0.0, local(z)],
            mass=mass,
            friction=friction,
            condim=condim,
            **extra,
        )

    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    collider(
        "container",
        cylinder,
        [shape.body_diameter / 2, shape.body_height / 2, 0.0],
        shape.body_height / 2,
        body_mass,
    )
    if shape.shoulder_height > 0:
        _add_shoulder(spec, container, bottle, local, shoulder_mass, friction, condim)
    neck_bottom = shape.body_height + shape.shoulder_height
    collider(
        "container_neck",
        cylinder,
        [shape.neck_diameter / 2, shape.neck_height / 2, 0.0],
        neck_bottom + shape.neck_height / 2,
        neck_mass,
    )
    if shape.support_ring_diameter > 0:
        collider(
            "container_ring",
            cylinder,
            [shape.support_ring_diameter / 2, RING_THICKNESS / 2, 0.0],
            total - shape.support_ring_drop + RING_THICKNESS / 2,
            ring_mass,
        )
    if liquid_ml > 0:
        _add_liquid(container, bottle, liquid_ml, local)


def _add_shoulder(spec, container, bottle, local, mass, friction, condim) -> None:
    shape = bottle.shape
    angles = np.linspace(0.0, 2 * math.pi, SEGMENTS, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    low, high = shape.body_height, shape.body_height + shape.shoulder_height
    vertices = np.vstack(
        [
            np.column_stack([unit * shape.body_diameter / 2, np.full(SEGMENTS, local(low))]),
            np.column_stack([unit * shape.neck_diameter / 2, np.full(SEGMENTS, local(high))]),
        ]
    )
    mesh = spec.add_mesh(name="container_shoulder")
    mesh.uservert = vertices.flatten().tolist()
    container.add_geom(
        name="container_shoulder",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="container_shoulder",
        mass=mass,
        friction=friction,
        condim=condim,
    )


def _add_liquid(container, bottle, liquid_ml, local) -> None:
    shape = bottle.shape
    mass = liquid_ml * LIQUID_KG_PER_ML
    centroid = liquid_centroid_m(shape, liquid_ml)
    level = level_for_volume_m(shape, liquid_ml)
    radius = shape.body_diameter / 2 - shape.t_side
    depth = max(level - shape.t_base, 1e-6)
    sideways = mass * (3 * radius**2 + depth**2) / 12
    along = mass * radius**2 / 2
    liquid = container.add_body(name="container_liquid", pos=[0.0, 0.0, local(centroid)])
    liquid.mass = mass
    liquid.ipos = [0.0, 0.0, 0.0]
    liquid.inertia = [sideways, sideways, along]
    liquid.explicitinertial = 1
    container.add_geom(
        name="container_level",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[radius * LEVEL_INSET, depth / 2, 0.0],
        pos=[0.0, 0.0, local(shape.t_base + depth / 2)],
        mass=0.0,
        contype=0,
        conaffinity=0,
        group=VISUAL_GROUP,
        rgba=LEVEL_RGBA,
    )
