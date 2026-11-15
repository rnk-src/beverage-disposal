"""Material constants are data: each figure is a range with a source, one table per material."""

import dataclasses

import pytest
from support.paths import ROOT

from beverage_disposal.core.containers import PROVENANCE_KINDS, Provenance
from beverage_disposal.core.materials import MaterialSpec, Range, load_materials

MATERIALS = load_materials(ROOT / "configs" / "materials.toml")
PET = MATERIALS["PET"]
SOURCE = Provenance("a test", "2026-11-15", "assumed")
FIGURES = [f.name for f in dataclasses.fields(MaterialSpec) if f.name != "name"]

FIGURE = (
    "{ lo = 1.0, hi = 2.0, "
    'provenance = { source = "a test", date = "2026-11-15", kind = "assumed" } }'
)


def material_file(tmp_path, **replace):
    """A one-material file in which every figure is `FIGURE`, except the ones in `replace`."""
    lines = ["[PET]"] + [f"{name} = {replace.get(name, FIGURE)}" for name in FIGURES]
    path = tmp_path / "materials.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_file_holds_pet_named_after_its_table():
    assert set(MATERIALS) == {"PET"}
    assert PET.name == "PET"


def test_a_range_with_its_low_end_above_its_high_end_is_refused():
    with pytest.raises(ValueError, match="low end"):
        Range(2.0, 1.0, SOURCE)


def test_a_range_may_be_a_single_value():
    assert Range(1.0, 1.0, SOURCE).lo == 1.0


@pytest.mark.parametrize("name", FIGURES)
def test_every_figure_of_pet_has_a_source_and_a_known_kind(name):
    figure = getattr(PET, name)
    assert figure.provenance.source.strip()
    assert figure.provenance.kind in PROVENANCE_KINDS


def test_a_file_in_which_every_figure_is_given_loads(tmp_path):
    assert load_materials(material_file(tmp_path))["PET"].density_g_cm3.hi == 2.0


def test_a_figure_without_a_source_is_refused(tmp_path):
    bare = "{ lo = 1.0, hi = 2.0 }"
    with pytest.raises(ValueError, match="provenance"):
        load_materials(material_file(tmp_path, density_g_cm3=bare))


def test_a_figure_with_a_low_end_above_its_high_end_is_refused_on_loading(tmp_path):
    backwards = FIGURE.replace("lo = 1.0, hi = 2.0", "lo = 2.0, hi = 1.0")
    with pytest.raises(ValueError, match="low end"):
        load_materials(material_file(tmp_path, density_g_cm3=backwards))


def test_the_thinnest_wall_is_at_most_seventy_percent_of_the_thinnest_published_one():
    """The owner's rule (2B): the thin-wall bound sits under the thinnest wall we found, 0.10 mm.

    A bound above it would let a lighter real bottle look heavier, which can misbin.
    """
    assert PET.t_side_mm.lo <= 0.7 * 0.10 + 1e-12  # tolerance for the float product


def test_the_density_lies_between_amorphous_and_crystalline_pet():
    assert 1.33 <= PET.density_g_cm3.lo <= PET.density_g_cm3.hi <= 1.455


@pytest.mark.parametrize("name", ["cap_g", "label_g", "wet_g"])
def test_the_lightest_empty_bottle_has_no_cap_no_label_and_no_water_on_it(name):
    """The lightest bound must not count extras: they can be missing from a real empty bottle."""
    assert getattr(PET, name).lo == 0.0
