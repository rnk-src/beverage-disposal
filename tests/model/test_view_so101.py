"""The viewer script: a snapshot without a window, error messages, and handing off to the viewer."""

import importlib.util
import shutil
import subprocess
import sys

import numpy as np
from damage import edit
from PIL import Image
from so101 import SO101
from support.images import foreground_fraction
from support.paths import ROOT

SCRIPT = ROOT / "scripts" / "view_so101.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def load_script_module():
    spec = importlib.util.spec_from_file_location("view_so101", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_exits_cleanly_and_documents_the_snapshot_option():
    result = run("--help")
    assert result.returncode == 0, result.stderr
    assert "--snapshot" in result.stdout


def test_a_snapshot_is_a_png_of_the_requested_size_and_shows_the_arm(tmp_path):
    out = tmp_path / "arm.png"
    result = run("--snapshot", str(out), "--width", "320", "--height", "240")
    assert result.returncode == 0, result.stderr
    with Image.open(out) as picture:
        assert picture.format == "PNG"
        assert picture.size == (320, 240)
        assert foreground_fraction(np.asarray(picture.convert("RGB"))) > 0.01


def test_the_snapshot_uses_the_model_it_is_given(tmp_path):
    copy = tmp_path / "so101"
    shutil.copytree(SO101, copy)
    edit(copy, "so101_new_calib.xml", 'group="2"', 'group="5"')  # hides every visual geom
    out = tmp_path / "blank.png"
    result = run("--model", str(copy / "so101_new_calib.xml"), "--snapshot", str(out))
    assert result.returncode == 0, result.stderr
    with Image.open(out) as picture:
        assert foreground_fraction(np.asarray(picture.convert("RGB"))) == 0.0


def test_a_missing_model_is_reported_by_path_and_exits_with_one(tmp_path):
    missing = tmp_path / "nowhere.xml"
    result = run("--model", str(missing), "--snapshot", str(tmp_path / "x.png"))
    assert result.returncode == 1
    assert str(missing) in result.stderr
    assert "Traceback" not in result.stderr


def test_without_a_snapshot_the_loaded_model_is_handed_to_the_viewer_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handed = []
    module = load_script_module()
    code = module.main([], launch=lambda model, data: handed.append((model.nq, data.qpos.shape)))
    assert code == 0
    assert handed == [(6, (6,))]
    assert list(tmp_path.iterdir()) == []


def test_a_snapshot_never_opens_the_viewer(tmp_path):
    module = load_script_module()

    def fail(model, data):
        raise AssertionError("the viewer must not open for a snapshot")

    assert module.main(["--snapshot", str(tmp_path / "s.png")], launch=fail) == 0
    assert (tmp_path / "s.png").is_file()
