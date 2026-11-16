"""The bottle family: ranges per nominal size, loaded from data, and a seeded sampler.

Expected ranges are read straight from the TOML text with `tomllib`, not from the loader under test.
"""

import dataclasses
import functools
import tomllib

import numpy as np
import pytest
from support.paths import ROOT

from beverage_disposal.core.bottle_family import (
    Bottle,
    GateRecord,
    Span,
    load_family,
    sample_bottle,
)
from beverage_disposal.core.bottles import capacity_ml, total_height_m
from beverage_disposal.core.materials import load_materials

BOTTLES = ROOT / "configs" / "bottles.toml"
PET = load_materials(ROOT / "configs" / "materials.toml")["PET"]
RAW = tomllib.loads(BOTTLES.read_text())
SIZES = [
    "pet_200ml",
    "pet_250ml",
    "pet_330ml",
    "pet_400ml",
    "pet_500ml",
    "pet_650ml",
    "pet_800ml",
    "pet_1000ml",
    "pet_1500ml",
    "pet_2000ml",
]


@functools.cache
def family_of_the_file():
    return load_family(BOTTLES)


MM = 1e-3
GRAM = 1e-3


def broken_file(tmp_path, old, new):
    text = BOTTLES.read_text()
    assert old in text, f"{old!r} is not in the file: the edit would change nothing"
    path = tmp_path / "bottles.toml"
    path.write_text(text.replace(old, new, 1))
    return path


def test_the_file_holds_ten_sizes_in_rising_order_named_after_their_nominal_size():
    assert list(family_of_the_file().classes) == SIZES
    for name, size in family_of_the_file().classes.items():
        assert size.name == name
        assert name == f"pet_{int(size.nominal_ml)}ml"


def test_every_size_has_a_source_and_a_range_that_is_ordered():
    for size in family_of_the_file().classes.values():
        assert size.provenance.source
        for field in ("body_diameter", "body_height", "shoulder_height", "overfill"):
            span = getattr(size, field)
            assert isinstance(span, Span) and span.lo <= span.hi


def test_a_span_with_its_low_end_above_its_high_end_is_refused():
    with pytest.raises(ValueError, match="low end"):
        Span(2.0, 1.0)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('grasp_mode = "neck"', 'grasp_mode = "claw"', "grasp_mode"),
        ('status = "failed"', 'status = "maybe"', "status"),
        ('base_styles = ["flat"]', 'base_styles = ["round"]', "base_style"),
        ('base_styles = ["flat"]', "base_styles = []", "base_style"),
        ("grasp_below_ring = 0.004\n", "", "grasp_below_ring"),
        ('grasp_mode = "neck"', 'grasp_mode = "neck"\nextra = 1', "extra"),
    ],
)
def test_a_bad_file_is_refused(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        load_family(broken_file(tmp_path, old, new))


def test_the_same_seed_gives_the_same_bottle_and_two_seeds_differ():
    one = sample_bottle(family_of_the_file(), "pet_500ml", PET, np.random.default_rng(7))
    again = sample_bottle(family_of_the_file(), "pet_500ml", PET, np.random.default_rng(7))
    other = sample_bottle(family_of_the_file(), "pet_500ml", PET, np.random.default_rng(8))
    assert one == again
    assert one != other
    assert isinstance(one, Bottle)


def test_one_generator_gives_a_different_bottle_each_call():
    rng = np.random.default_rng(1)
    assert sample_bottle(family_of_the_file(), "pet_500ml", PET, rng) != sample_bottle(
        family_of_the_file(), "pet_500ml", PET, rng
    )


@pytest.mark.parametrize("size", SIZES)
def test_the_brim_capacity_is_the_nominal_size_times_an_overfill_inside_its_range(size):
    nominal, over = RAW[size]["nominal_ml"], RAW[size]["overfill"]
    for seed in range(200):
        bottle = sample_bottle(family_of_the_file(), size, PET, np.random.default_rng(seed))
        capacity = capacity_ml(bottle.shape)
        assert nominal * over["lo"] * (1 - 1e-9) <= capacity <= nominal * over["hi"] * (1 + 1e-9)


@pytest.mark.parametrize("size", SIZES)
def test_every_drawn_dimension_lies_in_its_range_and_the_solved_body_height_too(size):
    raw = RAW[size]

    def inside(value, field):
        assert raw[field]["lo"] - 1e-12 <= value <= raw[field]["hi"] + 1e-12, field

    for seed in range(200):
        shape = sample_bottle(family_of_the_file(), size, PET, np.random.default_rng(seed)).shape
        inside(shape.body_diameter, "body_diameter")
        inside(shape.body_height, "body_height")
        inside(shape.shoulder_height, "shoulder_height")
        inside(shape.neck_diameter, "neck_diameter")
        inside(shape.neck_height, "neck_height")
        inside(shape.support_ring_diameter, "support_ring_diameter")
        inside(shape.support_ring_drop, "support_ring_drop")


def test_walls_finish_and_density_come_from_the_material_in_metres_and_kilograms():
    for seed in range(300):
        bottle = sample_bottle(family_of_the_file(), "pet_1000ml", PET, np.random.default_rng(seed))
        shape = bottle.shape
        assert PET.t_side_mm.lo * MM <= shape.t_side <= PET.t_side_mm.hi * MM
        assert PET.t_base_mm.lo * MM <= shape.t_base <= PET.t_base_mm.hi * MM
        assert PET.finish_g.lo * GRAM <= shape.finish_mass <= PET.finish_g.hi * GRAM
        assert 0.0 <= shape.cap_mass <= PET.cap_g.hi * GRAM
        assert 0.0 <= shape.label_mass <= PET.label_g.hi * GRAM
        assert PET.density_g_cm3.lo <= bottle.density_g_cm3 <= PET.density_g_cm3.hi


def test_a_bottle_may_lack_its_cap_or_its_label_and_often_has_both():
    bottles = [
        sample_bottle(family_of_the_file(), "pet_500ml", PET, np.random.default_rng(s)).shape
        for s in range(300)
    ]
    assert any(b.cap_mass == 0.0 for b in bottles) and any(b.cap_mass > 0 for b in bottles)
    assert any(b.label_mass == 0.0 for b in bottles) and any(b.label_mass > 0 for b in bottles)


def test_the_base_style_comes_from_the_sizes_own_list():
    small = {
        sample_bottle(
            family_of_the_file(), "pet_250ml", PET, np.random.default_rng(s)
        ).shape.base_style
        for s in range(100)
    }
    large = {
        sample_bottle(
            family_of_the_file(), "pet_2000ml", PET, np.random.default_rng(s)
        ).shape.base_style
        for s in range(200)
    }
    assert small == {"flat"}
    assert large == {"flat", "petaloid"}


def test_a_size_whose_body_height_range_cannot_hold_its_capacity_gives_up_and_names_it():
    squeezed = dataclasses.replace(
        family_of_the_file().classes["pet_500ml"], body_height=Span(0.5, 0.6)
    )
    family = dataclasses.replace(
        family_of_the_file(), classes={**family_of_the_file().classes, "pet_500ml": squeezed}
    )
    with pytest.raises(ValueError, match="pet_500ml"):
        sample_bottle(family, "pet_500ml", PET, np.random.default_rng(0))


def test_an_unknown_size_is_refused():
    with pytest.raises(ValueError, match="pet_3000ml"):
        sample_bottle(family_of_the_file(), "pet_3000ml", PET, np.random.default_rng(0))


def test_only_sizes_that_passed_the_gate_are_supported():
    assert family_of_the_file().supported() == ()
    passed = dataclasses.replace(
        family_of_the_file().classes["pet_330ml"], gate=GateRecord("passed", "2026-11-15", "")
    )
    failed = dataclasses.replace(
        family_of_the_file().classes["pet_500ml"],
        gate=GateRecord("failed", "2026-11-15", "out of reach"),
    )
    family = dataclasses.replace(
        family_of_the_file(),
        classes={**family_of_the_file().classes, "pet_330ml": passed, "pet_500ml": failed},
    )
    assert family.supported() == ("pet_330ml",)


def test_the_sampled_bottle_knows_its_size_and_empty_mass():
    bottle = sample_bottle(family_of_the_file(), "pet_500ml", PET, np.random.default_rng(3))
    assert bottle.size == "pet_500ml"
    assert bottle.empty_mass_kg > 0
    assert total_height_m(bottle.shape) > 0.15
