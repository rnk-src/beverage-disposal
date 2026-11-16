"""The classifier's feature vector: fixed names and order, and every number worked by hand."""

import math

import numpy as np
import pytest
from support.paths import ROOT

from beverage_disposal.core.bottle_family import load_family
from beverage_disposal.core.bottles import BottleShape, volume_below_ml
from beverage_disposal.core.bounds import EmptyMassBounds
from beverage_disposal.core.features import (
    BASE_FEATURES,
    MAY_BE_MISSING,
    feature_names,
    feature_vector,
    level_volume_bounds_ml,
)
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.pad import Weighing
from beverage_disposal.core.perception import DimensionEstimate, Perception, VisualEvidence

FAMILY = load_family(ROOT / "configs" / "bottles.toml")
PET = load_materials(ROOT / "configs" / "materials.toml")["PET"]


def seen(level=(0.010, 0.030), cap=True, base="petaloid", confidence=0.9, diameter=(0.060, 0.070)):
    dims = DimensionEstimate(
        body_diameter=diameter,
        total_height=(0.20, 0.22),
        shoulder_height=(0.03, 0.05),
        neck_diameter=(0.024, 0.028),
        neck_height=(0.018, 0.022),
    )
    return Perception(dims, VisualEvidence(level, cap, base, 0.9), confidence, "PET")


BOUNDS = EmptyMassBounds(min_kg=0.005, max_kg=0.045)
WEIGHING = Weighing(mass_kg=0.0321, stable=True)
V_BOUNDS = (0.0, 30.6)


def vector(perception=None):
    return feature_vector(perception or seen(), WEIGHING, BOUNDS, V_BOUNDS, FAMILY, PET)


def named(perception=None):
    return dict(zip(feature_names(FAMILY), vector(perception), strict=True))


def test_the_names_are_fixed_base_features_then_one_flag_per_size_class():
    assert feature_names(FAMILY) == (
        "d_lo",
        "d_hi",
        "h_lo",
        "h_hi",
        "s_lo",
        "s_hi",
        "neck_lo",
        "neck_hi",
        "w_g",
        "emin_g",
        "emax_g",
        "v_lo",
        "v_hi",
        "level_seen",
        "level_v_lo",
        "level_v_hi",
        "cap_seen",
        "base_petaloid",
        "base_confidence",
        "confidence",
        "size_pet_200ml",
        "size_pet_250ml",
        "size_pet_330ml",
        "size_pet_400ml",
        "size_pet_500ml",
        "size_pet_650ml",
        "size_pet_800ml",
        "size_pet_1000ml",
        "size_pet_1500ml",
        "size_pet_2000ml",
    )
    assert feature_names(FAMILY)[:20] == BASE_FEATURES
    assert set(MAY_BE_MISSING) <= set(BASE_FEATURES)


def test_the_vector_is_float64_and_as_long_as_the_names():
    assert vector().dtype == np.float64 and vector().shape == (30,)


def test_sizes_weight_bounds_and_confidence_are_copied_in_their_units():
    found = named()
    assert (found["d_lo"], found["d_hi"], found["h_lo"], found["h_hi"]) == (
        0.060,
        0.070,
        0.20,
        0.22,
    )
    assert (found["s_lo"], found["s_hi"], found["neck_lo"], found["neck_hi"]) == (
        0.03,
        0.05,
        0.024,
        0.028,
    )
    assert found["w_g"] == pytest.approx(32.1)
    assert (found["emin_g"], found["emax_g"]) == pytest.approx((5.0, 45.0))
    assert (found["v_lo"], found["v_hi"]) == (0.0, 30.6)
    assert (found["base_confidence"], found["confidence"]) == (0.9, 0.9)


def test_the_level_becomes_a_volume_on_the_nominal_bottle():
    # The middles: 65 mm wide, 210 mm tall, 40 mm shoulder, 26 mm neck 20 mm tall, so a 150 mm
    # body; PET walls 0.07 to 0.40 mm (middle 0.235) and base 0.15 to 1.20 mm (middle 0.675).
    nominal = BottleShape(0.065, 0.15, 0.04, 0.026, 0.02, "flat", 0.235e-3, 0.675e-3, 0, 0, 0)
    expected = (volume_below_ml(nominal, 0.010), volume_below_ml(nominal, 0.030))
    assert level_volume_bounds_ml(seen(), PET) == pytest.approx(expected)
    # By hand for the low end: a cylinder of inner radius 32.265 mm from the floor at 0.675 mm.
    by_hand = math.pi * 0.032265**2 * (0.010 - 0.000675) * 1e6
    assert level_volume_bounds_ml(seen(), PET)[0] == pytest.approx(by_hand)
    found = named()
    assert found["level_seen"] == 1.0
    assert (found["level_v_lo"], found["level_v_hi"]) == pytest.approx(expected)


def test_an_unseen_level_is_missing_not_zero():
    found = named(seen(level=None))
    assert found["level_seen"] == 0.0
    assert math.isnan(found["level_v_lo"]) and math.isnan(found["level_v_hi"])
    assert level_volume_bounds_ml(seen(level=None), PET) is None
    others = [value for name, value in found.items() if name not in MAY_BE_MISSING]
    assert not np.isnan(others).any()


def test_cap_and_base_are_flags_and_unknown_is_missing():
    assert (named()["cap_seen"], named()["base_petaloid"]) == (1.0, 1.0)
    flat = named(seen(cap=False, base="flat"))
    assert (flat["cap_seen"], flat["base_petaloid"]) == (0.0, 0.0)
    unknown = named(seen(cap=None, base=None))
    assert math.isnan(unknown["cap_seen"]) and math.isnan(unknown["base_petaloid"])


def test_the_size_flags_mark_every_class_the_measured_size_could_be():
    # 60 to 70 mm wide and 200 to 220 mm tall: the widths of 330 mL (55 to 63), 400 mL (58 to 66),
    # 500 mL (62 to 70), 650 mL (66 to 76) and 800 mL (70 to 82, touching) overlap, and each of
    # those classes can be that tall.
    found = named()
    flags = {name: value for name, value in found.items() if name.startswith("size_")}
    assert flags == {
        "size_pet_200ml": 0.0,
        "size_pet_250ml": 0.0,
        "size_pet_330ml": 1.0,
        "size_pet_400ml": 1.0,
        "size_pet_500ml": 1.0,
        "size_pet_650ml": 1.0,
        "size_pet_800ml": 1.0,
        "size_pet_1000ml": 0.0,
        "size_pet_1500ml": 0.0,
        "size_pet_2000ml": 0.0,
    }


def test_measured_sizes_that_make_no_bottle_give_no_level_volume():
    # A neck measured wider than the body cannot be turned into a nominal bottle.
    odd = seen(diameter=(0.010, 0.012))
    assert level_volume_bounds_ml(odd, PET) is None
    assert named(odd)["level_seen"] == 1.0 and math.isnan(named(odd)["level_v_hi"])
