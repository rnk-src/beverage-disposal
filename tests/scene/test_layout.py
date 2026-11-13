"""Where things stand on the table: footprints, the spawn box and the set-aside area."""

import dataclasses
import itertools

import numpy as np
import pytest
from support.paths import ROOT

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.layout import footprints, in_set_aside, sample_spawn

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
# Room kept between two footprints so a jaw can come down between them (assumed, 15 mm).
MIN_GAP = 0.015
CAN_RADIUS = 0.033


def grown(rect, by):
    xmin, xmax, ymin, ymax = rect
    return (xmin - by, xmax + by, ymin - by, ymax + by)


def gap(a, b):
    """Shortest distance between two axis-aligned rectangles; negative when they overlap."""
    dx = max(a[0] - b[1], b[0] - a[1])
    dy = max(a[2] - b[3], b[2] - a[3])
    return max(dx, dy) if max(dx, dy) < 0 else float(np.hypot(max(dx, 0), max(dy, 0)))


def test_the_footprints_are_the_outer_edges_of_what_the_file_describes():
    # By hand from configs/scene.toml: the bin's outer edge is its inner half-width plus a wall.
    rects = footprints(CONFIG)
    inner, wall = 0.10, 0.01
    cx, cy = CONFIG.bin.center_x, CONFIG.bin.center_y
    half = inner / 2 + wall
    assert rects["bin"] == pytest.approx((cx - half, cx + half, cy - half, cy + half))
    assert rects["pad"][1] - rects["pad"][0] == pytest.approx(2 * CONFIG.pad.half_x)
    assert set(rects) == {"bin", "pad", "set_aside", "spawn"}


def test_the_bin_the_pad_the_set_aside_area_and_the_spawn_box_keep_apart():
    # The spawn box is grown by the can's radius: a can at its edge must clear its neighbours.
    rects = footprints(CONFIG)
    rects["spawn"] = grown(rects["spawn"], CAN_RADIUS)
    for (a, ra), (b, rb) in itertools.combinations(rects.items(), 2):
        assert gap(ra, rb) >= MIN_GAP, f"{a} and {b} are {gap(ra, rb) * 1000:.0f} mm apart"


def test_everything_stands_on_the_table():
    table = CONFIG.table
    top = (
        table.center_x - table.half_x,
        table.center_x + table.half_x,
        -table.half_y,
        table.half_y,
    )
    for name, (xmin, xmax, ymin, ymax) in footprints(CONFIG).items():
        assert top[0] <= xmin and xmax <= top[1] and top[2] <= ymin and ymax <= top[3], name


# The spawn box straight from the file's numbers (centre 0.27, 0.0 and half width 0.04), so these
# tests do not lean on the footprints function they sit beside.
SPAWN_BOX = (0.23, 0.31, -0.04, 0.04)


def test_the_nominal_can_position_is_inside_the_spawn_box():
    xmin, xmax, ymin, ymax = SPAWN_BOX
    assert xmin <= CONFIG.container_x <= xmax and ymin <= CONFIG.container_y <= ymax
    assert footprints(CONFIG)["spawn"] == pytest.approx(SPAWN_BOX)


def test_spawn_positions_stay_in_the_box_and_use_all_of_it():
    rng = np.random.default_rng(0)
    points = np.array([sample_spawn(CONFIG, rng) for _ in range(2000)])
    xmin, xmax, ymin, ymax = SPAWN_BOX
    assert (points[:, 0] >= xmin).all() and (points[:, 0] <= xmax).all()
    assert (points[:, 1] >= ymin).all() and (points[:, 1] <= ymax).all()
    # Uniform draws land within 5 % of each edge at 2000 samples (miss chance about 1e-44).
    assert points[:, 0].min() < xmin + 0.05 * (xmax - xmin)
    assert points[:, 0].max() > xmax - 0.05 * (xmax - xmin)
    assert points[:, 1].min() < ymin + 0.05 * (ymax - ymin)
    assert points[:, 1].max() > ymax - 0.05 * (ymax - ymin)


def test_the_same_seed_gives_the_same_spawn_positions_and_another_seed_differs():
    def draw(seed):
        return [sample_spawn(CONFIG, np.random.default_rng(seed)) for _ in range(3)]

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)


def test_a_can_is_in_the_set_aside_area_only_when_its_whole_footprint_is():
    area = CONFIG.set_aside
    inside_limit = area.half_x - CAN_RADIUS
    assert in_set_aside(CONFIG, area.center_x, area.center_y)
    assert in_set_aside(CONFIG, area.center_x + inside_limit - 1e-6, area.center_y)
    assert not in_set_aside(CONFIG, area.center_x + inside_limit + 1e-6, area.center_y)
    assert not in_set_aside(CONFIG, area.center_x, area.center_y + area.half_y)


def test_the_set_aside_area_is_big_enough_for_a_can():
    assert CONFIG.set_aside.half_x > CAN_RADIUS and CONFIG.set_aside.half_y > CAN_RADIUS


def test_moving_the_bin_into_the_pad_breaks_the_keep_apart_rule():
    # Proof the rule can fail: the same check on a layout that is wrong.
    moved = dataclasses.replace(
        CONFIG,
        bin=dataclasses.replace(
            CONFIG.bin, center_x=CONFIG.pad.center_x, center_y=CONFIG.pad.center_y
        ),
    )
    rects = footprints(moved)
    assert gap(rects["bin"], rects["pad"]) < MIN_GAP
