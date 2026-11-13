"""The liquid wrench of a tilted rigid container: how the liquid loads a grasp.

Model M, unvalidated: a rigid cylinder with a closed bottom and an open top; the liquid is at rest
with a horizontal free surface (no sloshing, no surface tension, no foam); the container tilts
about a horizontal axis. In the container frame z runs up the axis from the bottom and x lies in
the tilt plane; the world's up direction there is (sin t, 0, cos t) for a tilt t.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np

G = 9.81  # m/s^2
NODES = 4000  # midpoint rule over the container's width; the volume is good to about 1e-6
MAX_TILT = math.radians(89.0)
BISECTIONS = 60


@dataclass(frozen=True)
class StaticsWrench:
    torque_nm: (
        float  # liquid's torque about the grasp axis (horizontal, perpendicular to the tilt plane)
    )
    liquid_mass_kg: float  # the liquid that stays in
    retained_ml: float
    spilled_ml: float


def _columns(radius: float, height: float, tilt: float, level: float):
    """Liquid column height at each x for a free surface x sin t + z cos t = level."""
    x = (np.arange(NODES) + 0.5) * (2 * radius / NODES) - radius
    width = 2 * np.sqrt(np.maximum(radius**2 - x**2, 0.0))
    column = np.clip((level - x * math.sin(tilt)) / math.cos(tilt), 0.0, height)
    return x, width, column, 2 * radius / NODES


def _volume(radius: float, height: float, tilt: float, level: float) -> float:
    _, width, column, dx = _columns(radius, height, tilt, level)
    return float((column * width).sum() * dx)


def liquid_wrench(
    radius: float,
    height: float,
    grasp_height: float,
    fill_ml: float,
    tilt: float,
    density: float = 1.0,
) -> StaticsWrench:
    """The torque of `fill_ml` of liquid (g/mL `density`) about the grasp axis at tilt `tilt` rad.

    Lengths in metres. Whatever does not fit below the low rim point spills. The torque is the
    liquid's weight times the horizontal distance from the grasp axis to its centroid, positive
    towards the side the container's top leans over.
    """
    if fill_ml < 0:
        raise ValueError(f"fill must not be negative, got {fill_ml} mL")
    if abs(tilt) >= MAX_TILT:
        raise ValueError(f"tilt must be under 89 degrees, got {math.degrees(tilt):.1f}")
    if fill_ml == 0:
        return StaticsWrench(0.0, 0.0, 0.0, 0.0)
    sign = 1.0 if tilt >= 0 else -1.0
    tilt = abs(tilt)
    wanted = fill_ml * 1e-6
    # The most that fits: the free surface touching the low rim point (x = -R, z = H).
    rim_level = -radius * math.sin(tilt) + height * math.cos(tilt)
    retained = min(wanted, _volume(radius, height, tilt, rim_level))
    low, high = -radius * math.sin(tilt), height * math.cos(tilt) + radius * math.sin(tilt)
    for _ in range(BISECTIONS):
        middle = (low + high) / 2
        if _volume(radius, height, tilt, middle) < retained:
            low = middle
        else:
            high = middle
    x, width, column, dx = _columns(radius, height, tilt, (low + high) / 2)
    volume = (column * width).sum() * dx
    x_centre = (x * column * width).sum() * dx / volume
    z_centre = (0.5 * column**2 * width).sum() * dx / volume
    # Horizontal world distance from the grasp axis to the centroid (e_h = (cos t, 0, -sin t)).
    lever = x_centre * math.cos(tilt) - (z_centre - grasp_height) * math.sin(tilt)
    mass = retained * density * 1e3
    return StaticsWrench(
        torque_nm=sign * mass * G * lever,
        liquid_mass_kg=mass,
        retained_ml=retained * 1e6,
        spilled_ml=(wanted - retained) * 1e6,
    )


def worst_case_torque_error(
    radius: float,
    height: float,
    grasp_height: float,
    fill_ml: float,
    tilt: float,
    density: float,
    *,
    radius_error: float,
    height_error: float,
    grasp_error: float,
    density_error: float,
    tilt_error: float,
) -> float:
    """The largest change in torque, N m, over the corners of the box of parameter errors.

    Each parameter is moved by plus or minus its error, every combination is tried, and the
    biggest departure from the nominal torque is returned. The torque is smooth in all five, so a
    corner is where the extreme lies for errors this small.
    """
    nominal = liquid_wrench(radius, height, grasp_height, fill_ml, tilt, density).torque_nm
    worst = 0.0
    for signs in itertools.product((-1.0, 1.0), repeat=5):
        r, h, g, rho, t = (
            radius + signs[0] * radius_error,
            height + signs[1] * height_error,
            grasp_height + signs[2] * grasp_error,
            density + signs[3] * density_error,
            tilt + signs[4] * tilt_error,
        )
        torque = liquid_wrench(r, h, g, fill_ml, t, rho).torque_nm
        worst = max(worst, abs(torque - nominal))
    return worst
