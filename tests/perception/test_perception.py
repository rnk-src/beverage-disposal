"""The perception record, the exact perception used before the noise model, and class matching."""

import dataclasses

import numpy as np
import pytest
from support.paths import ROOT

from beverage_disposal.core.bottle_family import classes_matching, load_family, sample_bottle
from beverage_disposal.core.bottles import Bottle, BottleShape, level_for_volume_m
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.perception import DimensionEstimate, exact_perception

FAMILY = load_family(ROOT / "configs" / "bottles.toml")
PET = load_materials(ROOT / "configs" / "materials.toml")["PET"]
SHAPE = BottleShape(
    body_diameter=0.065,
    body_height=0.15,
    shoulder_height=0.04,
    neck_diameter=0.026,
    neck_height=0.02,
    base_style="petaloid",
    t_side=0.0002,
    t_base=0.0005,
    finish_mass=0.002,
    cap_mass=0.0015,
    label_mass=0.0,
)
BOTTLE = Bottle(SHAPE, 1.38, "pet_500ml")


def dims(**changes):
    values = {
        "body_diameter": (0.065, 0.065),
        "total_height": (0.21, 0.21),
        "shoulder_height": (0.04, 0.04),
        "neck_diameter": (0.026, 0.026),
        "neck_height": (0.02, 0.02),
    }
    return DimensionEstimate(**{**values, **changes})


def test_an_interval_with_its_low_end_above_its_high_end_is_refused():
    with pytest.raises(ValueError, match="neck_height"):
        dims(neck_height=(0.03, 0.02))
    dims(neck_height=(0.02, 0.02))  # a zero-width interval is fine


def test_the_exact_perception_holds_the_true_sizes_as_zero_width_intervals():
    seen = exact_perception(BOTTLE, 100.0)
    assert seen.dims == dims()
    assert seen.confidence == 1.0 and seen.material == "PET"


def test_the_exact_perception_sees_the_true_level_cap_and_base():
    seen = exact_perception(BOTTLE, 100.0)
    level = level_for_volume_m(SHAPE, 100.0)
    assert seen.visual.level_m == (level, level)
    assert seen.visual.cap_seen is True
    assert (seen.visual.base_style, seen.visual.base_confidence) == ("petaloid", 1.0)
    capless = dataclasses.replace(BOTTLE, shape=dataclasses.replace(SHAPE, cap_mass=0.0))
    assert exact_perception(capless, 0.0).visual.cap_seen is False


def test_the_material_is_what_the_caller_says_it_is():
    assert exact_perception(BOTTLE, 0.0, material="glass").material == "glass"


def midpoint(span):
    return ((span.lo + span.hi) / 2,) * 2


@pytest.mark.parametrize("name", list(FAMILY.classes))
def test_a_bottle_in_the_middle_of_a_class_matches_that_class_alone(name):
    size = FAMILY.classes[name]
    body, shoulder, neck = (
        midpoint(size.body_height)[0],
        midpoint(size.shoulder_height)[0],
        midpoint(size.neck_height)[0],
    )
    seen = dims(
        body_diameter=midpoint(size.body_diameter), total_height=(body + shoulder + neck,) * 2
    )
    assert classes_matching(FAMILY, seen) == (name,)


def test_a_bottle_20_cm_wide_matches_no_class():
    assert classes_matching(FAMILY, dims(body_diameter=(0.2, 0.2))) == ()


def test_a_bottle_too_tall_for_its_width_matches_no_class():
    # 54 mm wide is a 250 mL width; the tallest 250 mL bottle is 0.145 + 0.045 + 0.024 = 0.214 m.
    assert classes_matching(FAMILY, dims(body_diameter=(0.054,) * 2, total_height=(0.16,) * 2))
    assert not classes_matching(FAMILY, dims(body_diameter=(0.054,) * 2, total_height=(0.22,) * 2))


def test_an_interval_spanning_two_classes_matches_both():
    # 250 mL bodies are 50 to 58 mm wide and 330 mL ones 55 to 63 mm.
    wide = dims(body_diameter=(0.052, 0.060), total_height=(0.15, 0.16))
    assert classes_matching(FAMILY, wide) == ("pet_250ml", "pet_330ml")


def test_an_interval_that_only_touches_a_class_range_at_its_end_still_matches():
    touching = dims(body_diameter=(0.045, 0.050), total_height=(0.15, 0.15))
    assert classes_matching(FAMILY, touching) == ("pet_250ml",)


@pytest.mark.parametrize("name", list(FAMILY.classes))
def test_every_drawn_bottle_matches_its_own_class(name):
    for seed in range(50):
        bottle = sample_bottle(FAMILY, name, PET, np.random.default_rng(seed))
        assert name in classes_matching(FAMILY, exact_perception(bottle, 0.0).dims)
