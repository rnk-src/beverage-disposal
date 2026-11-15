"""The scene file, its loader and the container record: one home for sizes and masses."""

import dataclasses

import pytest
from support.paths import ROOT

from beverage_disposal.core.config import from_mapping, load_scene_config
from beverage_disposal.core.containers import ContainerSpec, Provenance, total_mass

SCENE = ROOT / "configs" / "scene.toml"


def scene_with(tmp_path, old, new):
    """Copy the scene and container files to `tmp_path`, editing `old` to `new` where it occurs."""
    found = False
    for name in ["scene.toml", "containers.toml"]:
        text = (SCENE.parent / name).read_text()
        found = found or old in text
        (tmp_path / name).write_text(text.replace(old, new))
    assert found, f"{old!r} is in neither config file: the edit would change nothing"
    return tmp_path / "scene.toml"


def test_the_scene_names_the_can_and_gets_its_figures_from_the_registry():
    # 66 mm by 122 mm and 14 g are the figures in CLAUDE.md, none checked on a real can.
    can = load_scene_config(SCENE).container
    assert (can.diameter, can.height, can.empty_mass) == (0.066, 0.122, 0.014)
    assert can.grasp_height == 0.10  # the pads take the can near its rim
    assert can.provenance.kind == "assumed"
    assert can.provenance.source and can.provenance.date


def test_a_key_the_loader_does_not_know_is_rejected_and_named(tmp_path):
    path = scene_with(tmp_path, "thickness = 0.04", "thickness = 0.04\nwidth = 1.0")
    with pytest.raises(ValueError, match="width"):
        load_scene_config(path)


def test_a_missing_key_is_rejected_and_named(tmp_path):
    path = scene_with(tmp_path, "empty_mass = 0.014\n", "")
    with pytest.raises(ValueError, match="empty_mass"):
        load_scene_config(path)


def test_a_value_of_the_wrong_type_is_rejected_and_named(tmp_path):
    path = scene_with(tmp_path, "diameter = 0.066", 'diameter = "wide"')
    with pytest.raises(ValueError, match="diameter"):
        load_scene_config(path)


def test_a_whole_number_is_accepted_where_a_float_is_expected(tmp_path):
    path = scene_with(tmp_path, "container_y = 0.0", "container_y = 0")
    value = load_scene_config(path).container_y
    assert value == 0.0
    assert isinstance(value, float)


def test_an_unknown_provenance_kind_is_rejected(tmp_path):
    path = scene_with(tmp_path, 'kind = "assumed"', 'kind = "guessed"')
    with pytest.raises(ValueError, match="kind"):
        load_scene_config(path)


def test_a_loaded_config_cannot_be_changed():
    config = load_scene_config(SCENE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.container.diameter = 0.1


def test_the_contents_weigh_one_gram_per_millilitre():
    # A full 12 fl oz can holds 355 mL; 14 g of aluminium plus 355 g of water is 369 g.
    can = load_scene_config(SCENE).container
    assert total_mass(can, 0.0) == pytest.approx(0.014, abs=1e-12)
    assert total_mass(can, 355.0) == pytest.approx(0.369, abs=1e-12)


def test_a_negative_fill_is_rejected():
    can = ContainerSpec(
        name="x",
        diameter=0.05,
        height=0.1,
        empty_mass=0.01,
        grasp_height=0.05,
        friction=1.0,
        torsional_friction=0.01,
        provenance=Provenance(source="test", date="2026-11-13", kind="assumed"),
    )
    with pytest.raises(ValueError, match="fill"):
        total_mass(can, -1.0)


@dataclasses.dataclass(frozen=True)
class _Styles:
    names: tuple[str, ...]


def test_a_toml_list_fills_a_tuple_field_of_the_right_element_type():
    assert from_mapping(_Styles, {"names": ["a", "b"]}).names == ("a", "b")


@pytest.mark.parametrize("value", ["ab", ["a", 2], 3])
def test_a_tuple_field_refuses_a_non_list_or_a_wrong_element(value):
    with pytest.raises(ValueError, match="names"):
        from_mapping(_Styles, {"names": value})
