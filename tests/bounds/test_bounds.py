"""The empty-mass bounds: they bracket every drawn bottle, and a worked example by hand."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support.paths import ROOT

from beverage_disposal.core.bottle_family import load_family, sample_bottle
from beverage_disposal.core.bottles import areas
from beverage_disposal.core.bounds import empty_mass_bounds, shoulder_area_bounds_m2
from beverage_disposal.core.materials import load_materials, with_side_wall
from beverage_disposal.core.perception import DimensionEstimate, exact_perception

FAMILY = load_family(ROOT / "configs" / "bottles.toml")
PET = load_materials(ROOT / "configs" / "materials.toml")["PET"]
SIZES = list(FAMILY.classes)


def test_a_500_ml_bottle_worked_by_hand():
    # D 66 mm, total 210 mm, shoulder 40 mm, neck 26 mm wide and 20 mm tall: body 150 mm.
    dims = DimensionEstimate(
        (0.066, 0.066), (0.21, 0.21), (0.04, 0.04), (0.026, 0.026), (0.02, 0.02)
    )
    side = math.pi * 0.066 * 0.15
    base = math.pi * 0.033**2
    # Lower shoulder bound: the larger of the flat ring pi (R^2 - r^2) and pi (R + r) s.
    shoulder_lo = max(math.pi * (0.033**2 - 0.013**2), math.pi * (0.033 + 0.013) * 0.04)
    # Upper: the slant is at most the height plus the step in radius.
    shoulder_hi = math.pi * (0.033 + 0.013) * (0.04 + 0.033 - 0.013)
    # PET file: density 1.33 to 1.40 g/cm^3, side 0.07 to 0.40 mm, base 0.15 to 1.20 mm,
    # finish 1.2 to 4.0 g, cap up to 3.5 g, label up to 1.5 g, wet up to 5.0 g.
    lightest = 1330 * ((side + shoulder_lo) * 0.07e-3 + base * 0.15e-3) + 1.2e-3
    heaviest = (
        1400 * ((side + shoulder_hi) * 0.40e-3 + base * 1.20e-3) + (4.0 + 3.5 + 1.5 + 5.0) * 1e-3
    )
    bounds = empty_mass_bounds(dims, PET)
    assert bounds.min_kg == pytest.approx(lightest, rel=1e-12)
    assert bounds.max_kg == pytest.approx(heaviest, rel=1e-12)
    # A sanity anchor: a few grams to a few tens of grams.
    assert 0.004 < bounds.min_kg < 0.006 and 0.03 < bounds.max_kg < 0.05


@pytest.mark.parametrize("size", SIZES)
def test_every_drawn_bottle_weighs_between_its_bounds_dry_or_wet(size):
    wet_hi = PET.wet_g.hi * 1e-3
    for seed in range(1700):  # 10 200 bottles over the six sizes
        bottle = sample_bottle(FAMILY, size, PET, np.random.default_rng(seed))
        bounds = empty_mass_bounds(exact_perception(bottle, 0.0).dims, PET)
        assert bounds.min_kg <= bottle.empty_mass_kg
        assert bottle.empty_mass_kg + wet_hi <= bounds.max_kg


lengths = st.floats(0.0, 0.004)


@settings(max_examples=200, derandomize=True, deadline=None)
@given(seed=st.integers(0, 10_000), size=st.sampled_from(SIZES), grow=st.tuples(*[lengths] * 10))
def test_widening_any_interval_never_raises_the_minimum_or_lowers_the_maximum(seed, size, grow):
    bottle = sample_bottle(FAMILY, size, PET, np.random.default_rng(seed))
    exact = exact_perception(bottle, 0.0).dims
    names = ["body_diameter", "total_height", "shoulder_height", "neck_diameter", "neck_height"]
    wider = DimensionEstimate(
        **{
            name: (
                max(0.0, getattr(exact, name)[0] - grow[2 * i]),
                getattr(exact, name)[1] + grow[2 * i + 1],
            )
            for i, name in enumerate(names)
        }
    )
    tight, loose = empty_mass_bounds(exact, PET), empty_mass_bounds(wider, PET)
    assert loose.min_kg <= tight.min_kg + 1e-15
    assert loose.max_kg >= tight.max_kg - 1e-15


def test_the_shoulder_bounds_bracket_the_true_area_of_random_shapes():
    rng = np.random.default_rng(0)
    for _ in range(10_000):
        small = rng.uniform(0.005, 0.02)
        big = small + rng.uniform(0.0, 0.05)
        height = rng.uniform(0.0, 0.1)
        true = math.pi * (big + small) * math.hypot(height, big - small)
        # Intervals that contain the true radii and height, each up to 3 mm wide on each side.
        pad = rng.uniform(0.0, 0.003, 6)
        low, high = shoulder_area_bounds_m2(
            (big - pad[0], big + pad[1]),
            (max(0.0, small - pad[2]), small + pad[3]),
            (max(0.0, height - pad[4]), height + pad[5]),
        )
        assert low <= true * (1 + 1e-12) and true <= high * (1 + 1e-12)
        assert low >= 0.0


def test_the_shoulder_bounds_agree_with_the_shape_code_on_a_drawn_bottle():
    bottle = sample_bottle(FAMILY, "pet_1000ml", PET, np.random.default_rng(3))
    shape = bottle.shape
    radii = ((shape.body_diameter / 2,) * 2, (shape.neck_diameter / 2,) * 2)
    low, high = shoulder_area_bounds_m2(*radii, (shape.shoulder_height,) * 2)
    assert low <= areas(shape).shoulder <= high


def test_a_height_interval_that_leaves_no_room_for_a_body_is_refused():
    dims = DimensionEstimate((0.066,) * 2, (0.05, 0.05), (0.04,) * 2, (0.026,) * 2, (0.02,) * 2)
    with pytest.raises(ValueError, match="body"):
        empty_mass_bounds(dims, PET)


def narrowed_by(bottle, half):
    """PET with its wall narrowed to a reading that holds the bottle's true wall, in mm."""
    t = bottle.shape.t_side * 1000
    return with_side_wall(PET, (t / (1 + half), t / (1 - half)))


@pytest.mark.parametrize("size", SIZES)
def test_every_drawn_bottle_weighs_between_its_bounds_with_a_true_wall_reading(size):
    wet_hi = PET.wet_g.hi * 1e-3
    for seed in range(300):
        bottle = sample_bottle(FAMILY, size, PET, np.random.default_rng(seed))
        narrowed = narrowed_by(bottle, 0.10)
        bounds = empty_mass_bounds(exact_perception(bottle, 0.0).dims, narrowed)
        assert bounds.min_kg <= bottle.empty_mass_kg
        assert bottle.empty_mass_kg + wet_hi <= bounds.max_kg


@settings(max_examples=200, derandomize=True, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    size=st.sampled_from(SIZES),
    half=st.floats(0.0, 0.9),
)
def test_a_wall_reading_never_lowers_the_minimum_or_raises_the_maximum(seed, size, half):
    bottle = sample_bottle(FAMILY, size, PET, np.random.default_rng(seed))
    dims = exact_perception(bottle, 0.0).dims
    wide, narrow = empty_mass_bounds(dims, PET), empty_mass_bounds(dims, narrowed_by(bottle, half))
    assert narrow.min_kg >= wide.min_kg
    assert narrow.max_kg <= wide.max_kg


def test_a_2000_ml_bottle_with_a_measured_wall_worked_by_hand():
    # D 105 mm, total 335 mm, shoulder 60 mm, neck 30 mm wide and 20 mm tall: body 255 mm.
    dims = DimensionEstimate(
        (0.105, 0.105), (0.335, 0.335), (0.06, 0.06), (0.03, 0.03), (0.02, 0.02)
    )
    big, small = 0.0525, 0.015
    side = math.pi * 0.105 * 0.255
    base = math.pi * big**2
    shoulder_lo = max(math.pi * (big**2 - small**2), math.pi * (big + small) * 0.06)
    shoulder_hi = math.pi * (big + small) * (0.06 + big - small)
    # The wall is read as 0.20 to 0.30 mm, inside the file's 0.07 to 0.40 mm; the rest is as filed.
    wall = with_side_wall(PET, (0.20, 0.30))
    lightest = 1330 * ((side + shoulder_lo) * 0.20e-3 + base * 0.15e-3) + 1.2e-3
    heaviest = (
        1400 * ((side + shoulder_hi) * 0.30e-3 + base * 1.20e-3) + (4.0 + 3.5 + 1.5 + 5.0) * 1e-3
    )
    bounds = empty_mass_bounds(dims, wall)
    assert bounds.min_kg == pytest.approx(lightest, rel=1e-12)
    assert bounds.max_kg == pytest.approx(heaviest, rel=1e-12)
    wide = empty_mass_bounds(dims, PET)
    assert bounds.min_kg > wide.min_kg and bounds.max_kg < wide.max_kg
