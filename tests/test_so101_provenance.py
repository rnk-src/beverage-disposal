"""The record of where the vendored SO-101 files came from."""

import hashlib
import re
import tomllib
from pathlib import Path

from so101 import referenced_meshes

ROOT = Path(__file__).resolve().parents[1]
SO101 = ROOT / "assets" / "so101"


def load_provenance():
    return tomllib.loads((SO101 / "PROVENANCE.toml").read_text())


def test_provenance_names_the_upstream_source_license_and_calibration():
    record = load_provenance()
    assert record["upstream"]["repository"] == "https://github.com/TheRobotStudio/SO-ARM100"
    assert record["upstream"]["path"] == "Simulation/SO101"
    assert re.fullmatch(r"[0-9a-f]{40}", record["upstream"]["commit"])
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(record["upstream"]["retrieved"]))
    assert record["license"]["spdx"] == "Apache-2.0"
    assert record["license"]["file"] == "LICENSE"
    assert record["model"]["calibration"] == "new"
    assert record["model"]["mjcf"] == "so101_new_calib.xml"
    assert record["model"]["urdf"] == "so101_new_calib.urdf"


# Deliberately not the checker's own hashing helper: this test is the independent second
# opinion on it.
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_provenance_lists_exactly_the_vendored_files_with_their_hashes():
    from_mjcf, _ = referenced_meshes(SO101)
    expected = {
        "so101_new_calib.xml",
        "so101_new_calib.urdf",
        "README.md",
        "LICENSE",
        *(str(path.relative_to(SO101)) for path in from_mjcf),
    }
    recorded = load_provenance()["files"]
    assert set(recorded) == expected
    assert {name: sha256(SO101 / name) for name in recorded} == recorded
