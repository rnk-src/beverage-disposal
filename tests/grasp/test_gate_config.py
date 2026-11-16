"""The gate's bar comes from its own file, and the episode runner gives a tall bottle room."""

import tomllib

import pytest
from support.paths import ROOT

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.gate import load_gate_bar

GATE = ROOT / "configs" / "gate.toml"
SCENE = load_scene_config(ROOT / "configs" / "scene.toml")


def test_the_bar_is_the_locked_one_2_mm_2_s_and_12_of_12():
    bar = load_gate_bar(GATE)
    assert (bar.max_slip_m, bar.max_swing_settle_s, bar.required_places) == (0.002, 2.0, 12)
    assert bar.seeds == 12
    assert bar.swing_band_rad == tomllib.loads(GATE.read_text())["swing_band_rad"]


def test_a_gate_file_with_a_missing_figure_is_refused(tmp_path):
    path = tmp_path / "gate.toml"
    path.write_text(GATE.read_text().replace("max_slip_m", "slip"))
    with pytest.raises(ValueError):
        load_gate_bar(path)
