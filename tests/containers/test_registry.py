"""The container registry: types are data, weights have sources, unknown types are enrolled."""

import dataclasses
import subprocess
import sys

import mujoco
import pytest
from support.paths import ROOT
from support.so101 import MJCF, SO101

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.containers import PROVENANCE_KINDS, total_mass
from beverage_disposal.core.registry import EnrolmentRequest, load_registry
from beverage_disposal.sim.scene import build_scene

CONFIGS = ROOT / "configs"
REGISTRY = load_registry(CONFIGS / "containers.toml")


def test_the_registry_holds_the_can_and_the_bottle():
    assert set(REGISTRY.records) == {"can_12floz", "bottle_500ml"}


def test_the_can_record_holds_the_figures_of_the_12_fl_oz_can():
    # 66 mm by 122 mm and 14 g are the figures in containers.toml; 13 g is the lightest empty can.
    can = REGISTRY.lookup("can_12floz")
    assert (can.spec.diameter, can.spec.height, can.spec.empty_mass) == (0.066, 0.122, 0.014)
    assert can.lightest_empty_mass == 0.013
    assert can.has_tab and not can.cap_check_required
    assert can.spec.grasp_height == 0.10  # the pads take the can near its rim
    assert can.spec.grasp_diameter == 0.066  # the jaws close on the whole can


def test_the_bottle_record_is_plain_data_with_a_cap_rule_and_cited_figures():
    bottle = REGISTRY.lookup("bottle_500ml")
    assert (bottle.spec.diameter, bottle.spec.height) == (0.0635, 0.203)
    assert bottle.spec.grasp_diameter == 0.0635
    assert bottle.lightest_empty_mass == 0.0185
    assert bottle.cap_check_required and not bottle.has_tab
    assert bottle.spec.provenance.kind == "cited" and bottle.weight_provenance.kind == "cited"


@pytest.mark.parametrize("type_id", ["can_12floz", "bottle_500ml"])
def test_every_record_says_where_its_numbers_came_from(type_id):
    record = REGISTRY.lookup(type_id)
    for provenance in (record.spec.provenance, record.weight_provenance):
        assert provenance.source and provenance.date
        assert provenance.kind in PROVENANCE_KINDS
    assert record.spec.name == record.type_id == type_id


@pytest.mark.parametrize("type_id", ["can_12floz", "bottle_500ml"])
def test_the_lightest_empty_weight_is_not_above_the_typical_empty_weight(type_id):
    record = REGISTRY.lookup(type_id)
    assert 0 < record.lightest_empty_mass <= record.spec.empty_mass


def test_a_known_type_is_found_and_makes_no_request():
    record, after = REGISTRY.meet("can_12floz", observed_mass=0.37)
    assert record is REGISTRY.lookup("can_12floz")
    assert after.pending == ()


def test_an_unknown_type_is_not_found_and_leaves_an_enrolment_request():
    record, after = REGISTRY.meet("juice_1l", observed_mass=0.35)
    assert record is None
    assert after.pending == (EnrolmentRequest(type_id="juice_1l", observed_mass=0.35),)
    assert REGISTRY.pending == ()  # the registry itself is never changed in place


def test_meeting_the_same_unknown_type_again_keeps_one_request_with_the_latest_weight():
    _, once = REGISTRY.meet("juice_1l", observed_mass=0.35)
    _, twice = once.meet("juice_1l", observed_mass=0.36)
    assert twice.pending == (EnrolmentRequest(type_id="juice_1l", observed_mass=0.36),)


def test_enrolling_adds_the_record_and_clears_its_request():
    _, waiting = REGISTRY.meet("can_12floz_slim", observed_mass=0.30)
    template = REGISTRY.lookup("can_12floz")
    slim = dataclasses.replace(
        template,
        type_id="can_12floz_slim",
        spec=dataclasses.replace(template.spec, name="can_12floz_slim", diameter=0.057),
    )
    enrolled = waiting.enrol(slim)
    assert enrolled.lookup("can_12floz_slim") is slim
    assert enrolled.pending == ()
    assert waiting.lookup("can_12floz_slim") is None


def test_enrolling_a_type_that_already_exists_is_refused():
    with pytest.raises(ValueError, match="can_12floz"):
        REGISTRY.enrol(REGISTRY.lookup("can_12floz"))


def test_a_registry_record_with_a_key_nobody_knows_is_rejected_and_named(tmp_path):
    text = (
        (CONFIGS / "containers.toml")
        .read_text()
        .replace('material = "aluminium"', 'material = "aluminium"\ncolour = "red"')
    )
    path = tmp_path / "containers.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="colour"):
        load_registry(path)


def test_a_record_whose_lightest_weight_is_above_its_typical_weight_is_refused():
    can = REGISTRY.lookup("can_12floz")
    with pytest.raises(ValueError, match="lightest"):
        dataclasses.replace(can, lightest_empty_mass=can.spec.empty_mass + 0.001)


def test_a_record_whose_spec_has_another_name_is_refused():
    can = REGISTRY.lookup("can_12floz")
    with pytest.raises(ValueError, match="named"):
        dataclasses.replace(can, spec=dataclasses.replace(can.spec, name="something_else"))


def test_a_second_container_type_needs_data_only_and_no_code(tmp_path):
    # A taller, wider can goes into the registry file and the scene names it: no code changes.
    extra = (CONFIGS / "containers.toml").read_text() + EXTRA_TYPE
    (tmp_path / "containers.toml").write_text(extra)
    scene = (CONFIGS / "scene.toml").read_text().replace('"can_12floz"', '"can_16floz"')
    (tmp_path / "scene.toml").write_text(scene)
    config = dataclasses.replace(load_scene_config(tmp_path / "scene.toml"), fill_ml=473.0)
    model = build_scene(config, SO101 / MJCF)
    geom = model.geom("container")
    assert geom.size[0] * 2 == pytest.approx(0.0683, abs=1e-9)
    assert geom.size[1] * 2 == pytest.approx(0.1627, abs=1e-9)
    assert model.body("container").mass[0] == pytest.approx(0.017 + 0.473, abs=1e-9)
    assert total_mass(config.container, 473.0) == pytest.approx(0.49, abs=1e-12)
    assert model.geom_type[geom.id] == mujoco.mjtGeom.mjGEOM_CYLINDER


def test_a_scene_naming_an_unknown_container_type_is_rejected_by_name(tmp_path):
    (tmp_path / "containers.toml").write_text((CONFIGS / "containers.toml").read_text())
    (tmp_path / "scene.toml").write_text(
        (CONFIGS / "scene.toml").read_text().replace('"can_12floz"', '"keg_50l"')
    )
    with pytest.raises(ValueError, match="keg_50l"):
        load_scene_config(tmp_path / "scene.toml")


EXTRA_TYPE = """
[can_16floz]
material = "aluminium"
lightest_empty_mass = 0.016
cap_check_required = false
has_tab = true

[can_16floz.weight_provenance]
source = "test data"
date = "2026-11-13"
kind = "assumed"

[can_16floz.spec]
name = "can_16floz"
diameter = 0.0683
height = 0.1627
grasp_diameter = 0.0683
empty_mass = 0.017
grasp_height = 0.14
friction = 1.0
torsional_friction = 0.01

[can_16floz.spec.provenance]
source = "test data"
date = "2026-11-13"
kind = "assumed"
"""


WITHOUT_MUJOCO = """
import sys
sys.modules["mujoco"] = None
from beverage_disposal.core.registry import load_registry
registry = load_registry(sys.argv[1])
print(sorted(registry.records))
"""


def test_the_registry_loads_with_mujoco_unavailable():
    result = subprocess.run(
        [sys.executable, "-c", WITHOUT_MUJOCO, str(CONFIGS / "containers.toml")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "can_12floz" in result.stdout
