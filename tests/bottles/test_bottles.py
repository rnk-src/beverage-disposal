"""Bottle geometry: areas, shell mass, capacity and the level of a liquid, against closed forms.

Every expected value is worked out here from the shape's numbers (metres, kilograms), not by the
code under test. The 3-4-5 triangle makes the shoulder's slant exactly 0.05 m.
"""

import dataclasses
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from beverage_disposal.core.bottles import (
    BottleShape,
    areas,
    capacity_ml,
    empty_mass_kg,
    level_for_volume_m,
    shell_mass_kg,
    total_height_m,
    volume_below_ml,
)

PI = math.pi
M3_TO_ML = 1e6

# Body 80 mm wide and 100 mm tall, a 40 mm shoulder narrowing to a 20 mm neck, a 30 mm neck.
CONE = BottleShape(
    body_diameter=0.08,
    body_height=0.10,
    shoulder_height=0.04,
    neck_diameter=0.02,
    neck_height=0.03,
    base_style="flat",
    t_side=0.0,
    t_base=0.0,
    finish_mass=0.0,
    cap_mass=0.0,
    label_mass=0.0,
)
# A 60 mm cylinder 150 mm tall, built as body + a shoulder that does not narrow + a neck.
CYLINDER = dataclasses.replace(
    CONE, body_diameter=0.06, body_height=0.10, shoulder_height=0.02, neck_diameter=0.06
)
WALLED = dataclasses.replace(CYLINDER, t_side=0.001, t_base=0.002)


def test_the_total_height_is_body_plus_shoulder_plus_neck():
    assert total_height_m(CONE) == pytest.approx(0.10 + 0.04 + 0.03)


def test_a_cylinder_built_from_three_parts_holds_pi_r_squared_h():
    expected = PI * 0.03**2 * 0.15 * M3_TO_ML
    assert capacity_ml(CYLINDER) == pytest.approx(expected, rel=1e-12)


def test_a_shape_with_a_conical_shoulder_holds_body_plus_frustum_plus_neck():
    body = PI * 0.04**2 * 0.10
    frustum = PI * 0.04 / 3 * (0.04**2 + 0.04 * 0.01 + 0.01**2)
    neck = PI * 0.01**2 * 0.03
    assert capacity_ml(CONE) == pytest.approx((body + frustum + neck) * M3_TO_ML, rel=1e-12)


def test_the_walls_take_room_from_the_inside_and_the_base_lifts_the_floor():
    # Inner radius 0.029 m; the floor is 0.002 m up, so the inner height is 0.148 m.
    expected = PI * 0.029**2 * 0.148 * M3_TO_ML
    assert capacity_ml(WALLED) == pytest.approx(expected, rel=1e-12)


def test_the_volume_below_a_level_in_the_body_is_a_cylinder_above_the_floor():
    level = 0.052  # 50 mm above the inner floor
    expected = PI * 0.029**2 * 0.050 * M3_TO_ML
    assert volume_below_ml(WALLED, level) == pytest.approx(expected, rel=1e-12)


def test_the_volume_below_a_level_inside_the_shoulder_adds_a_partial_frustum():
    # Level 20 mm up the 40 mm shoulder: the radius there is halfway, (0.04 + 0.01) / 2.
    level = 0.10 + 0.02
    radius_here = 0.025
    frustum = PI * 0.02 / 3 * (0.04**2 + 0.04 * radius_here + radius_here**2)
    expected = (PI * 0.04**2 * 0.10 + frustum) * M3_TO_ML
    assert volume_below_ml(CONE, level) == pytest.approx(expected, rel=1e-12)


def test_a_level_at_or_below_the_floor_holds_nothing_and_one_above_the_top_holds_the_capacity():
    assert volume_below_ml(WALLED, 0.002) == 0.0
    assert volume_below_ml(WALLED, 0.0) == 0.0
    assert volume_below_ml(WALLED, 1.0) == pytest.approx(capacity_ml(WALLED), rel=1e-12)


def test_the_level_of_no_liquid_is_the_floor_and_of_a_full_bottle_the_top():
    assert level_for_volume_m(WALLED, 0.0) == pytest.approx(0.002)
    assert level_for_volume_m(WALLED, capacity_ml(WALLED)) == pytest.approx(
        total_height_m(WALLED), abs=1e-9
    )


@settings(derandomize=True, max_examples=200)
@given(fraction=st.floats(min_value=0.0, max_value=1.0))
def test_the_level_for_a_volume_inverts_the_volume_below_a_level(fraction):
    shape = dataclasses.replace(CONE, t_side=0.001, t_base=0.002)
    level = 0.002 + fraction * (total_height_m(shape) - 0.002)
    assert level_for_volume_m(shape, volume_below_ml(shape, level)) == pytest.approx(
        level, abs=1e-9
    )


@settings(derandomize=True, max_examples=200)
@given(
    low=st.floats(min_value=0.0, max_value=0.17),
    high=st.floats(min_value=0.0, max_value=0.17),
)
def test_a_higher_level_never_holds_less(low, high):
    low, high = sorted((low, high))
    assert volume_below_ml(CONE, low) <= volume_below_ml(CONE, high) + 1e-9


@pytest.mark.parametrize("volume", [-1.0, 10_000.0])
def test_a_volume_below_zero_or_above_the_capacity_has_no_level(volume):
    with pytest.raises(ValueError, match="volume"):
        level_for_volume_m(CONE, volume)


def test_the_areas_are_side_base_and_the_shoulders_slanted_surface():
    found = areas(CONE)
    assert found.side == pytest.approx(PI * 0.08 * 0.10, rel=1e-12)
    assert found.base == pytest.approx(PI * 0.04**2, rel=1e-12)
    # Radii 0.04 and 0.01, height 0.04: slant sqrt(0.04^2 + 0.03^2) = 0.05; area pi (R + r) slant.
    assert found.shoulder == pytest.approx(PI * (0.04 + 0.01) * 0.05, rel=1e-12)


def test_the_shell_mass_is_density_times_each_area_times_its_thickness():
    shape = dataclasses.replace(CONE, t_side=0.0002, t_base=0.0005)
    side, base, shoulder = PI * 0.08 * 0.10, PI * 0.04**2, PI * 0.05 * 0.05
    expected = 1380.0 * (side * 0.0002 + base * 0.0005 + shoulder * 0.0002)
    assert shell_mass_kg(shape, density_g_cm3=1.38) == pytest.approx(expected, rel=1e-12)


def test_the_empty_mass_adds_the_neck_cap_and_label_to_the_shell():
    shape = dataclasses.replace(
        CONE, t_side=0.0002, t_base=0.0005, finish_mass=0.002, cap_mass=0.001, label_mass=0.0005
    )
    shell = shell_mass_kg(shape, density_g_cm3=1.38)
    assert empty_mass_kg(shape, density_g_cm3=1.38) == pytest.approx(shell + 0.0035, rel=1e-12)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"neck_diameter": 0.09}, "neck"),
        ({"body_diameter": 0.0}, "positive"),
        ({"body_height": -0.1}, "positive"),
        ({"neck_height": 0.0}, "positive"),
        ({"t_side": -0.001}, "negative"),
        ({"t_side": 0.011}, "wall"),  # thicker than the neck's radius of 0.01 m
        ({"t_base": 0.10}, "base"),  # as tall as the body
        ({"base_style": "round"}, "base_style"),
        ({"cap_mass": -0.001}, "negative"),
    ],
)
def test_an_impossible_shape_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(CONE, **change)
