"""Where the bin, the pad, the set-aside area and the spawn box stand, and what counts as inside."""

import numpy as np

from beverage_disposal.core.config import SceneConfig

# Rectangle on the table: x_min, x_max, y_min, y_max in the arm's base frame, metres.
Rect = tuple[float, float, float, float]


def _around(center_x: float, center_y: float, half_x: float, half_y: float) -> Rect:
    return (center_x - half_x, center_x + half_x, center_y - half_y, center_y + half_y)


def footprints(config: SceneConfig) -> dict[str, Rect]:
    """The table area each fixture covers; the bin's is its outer edge."""
    bin_ = config.bin
    return {
        "bin": _around(
            bin_.center_x,
            bin_.center_y,
            bin_.inner_x / 2 + bin_.wall,
            bin_.inner_y / 2 + bin_.wall,
        ),
        "pad": _around(
            config.pad.center_x, config.pad.center_y, config.pad.half_x, config.pad.half_y
        ),
        "set_aside": _around(
            config.set_aside.center_x,
            config.set_aside.center_y,
            config.set_aside.half_x,
            config.set_aside.half_y,
        ),
        "spawn": _around(
            config.spawn.center_x,
            config.spawn.center_y,
            config.spawn.half_size,
            config.spawn.half_size,
        ),
    }


def sample_spawn(config: SceneConfig, rng: np.random.Generator) -> tuple[float, float]:
    """A uniform random container position in the spawn box, metres."""
    x_min, x_max, y_min, y_max = footprints(config)["spawn"]
    return float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max))


def in_set_aside(config: SceneConfig, x: float, y: float) -> bool:
    """Whether a container centred at (x, y), metres, lies wholly inside the set-aside area."""
    radius = config.container.diameter / 2
    area = config.set_aside
    return abs(x - area.center_x) <= area.half_x - radius and abs(y - area.center_y) <= (
        area.half_y - radius
    )
