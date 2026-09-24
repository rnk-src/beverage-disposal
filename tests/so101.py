"""Reads facts out of the vendored SO-101 files, so tests can also point it at a damaged copy."""

import hashlib
import os
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

# SO101_DIR lets a test run the model tests against a damaged copy, to prove they can fail.
SO101 = Path(os.environ.get("SO101_DIR") or Path(__file__).resolve().parents[1] / "assets" / "so101")
MJCF = "so101_new_calib.xml"
URDF = "so101_new_calib.urdf"


def model_facts(root):
    model = mujoco.MjModel.from_xml_path(str(Path(root) / MJCF))

    def name(kind, index):
        return mujoco.mj_id2name(model, kind, index)

    joint = mujoco.mjtObj.mjOBJ_JOINT
    actuator = mujoco.mjtObj.mjOBJ_ACTUATOR
    joint_names = [name(joint, i) for i in range(model.njnt)]
    actuator_names = [name(actuator, i) for i in range(model.nu)]
    return {
        "nq": model.nq,
        "joint_names": joint_names,
        "joint_ranges": {n: tuple(model.jnt_range[i]) for i, n in enumerate(joint_names)},
        "actuator_names": actuator_names,
        "actuator_targets": {
            n: name(joint, model.actuator_trnid[i][0]) for i, n in enumerate(actuator_names)
        },
        "actuator_ctrlranges": {
            n: tuple(model.actuator_ctrlrange[i]) for i, n in enumerate(actuator_names)
        },
    }


def urdf_facts(root):
    joints = ET.parse(Path(root) / URDF).getroot().findall("joint")
    revolute = [j for j in joints if j.get("type") == "revolute"]
    return {
        "revolute": [j.get("name") for j in revolute],
        "fixed": [j.get("name") for j in joints if j.get("type") == "fixed"],
        "limits": {
            j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
            for j in revolute
        },
    }


def referenced_meshes(root):
    """Mesh files the MJCF and the URDF point at, each as a set of absolute paths."""
    root = Path(root)
    mjcf = ET.parse(root / MJCF).getroot()
    mesh_dir = root / mjcf.find("compiler").get("meshdir")
    from_mjcf = {mesh_dir / mesh.get("file") for mesh in mjcf.iter("mesh")}
    urdf = ET.parse(root / URDF).getroot()
    from_urdf = {root / mesh.get("filename") for mesh in urdf.iter("mesh")}
    return from_mjcf, from_urdf


REQUIRED_FIELDS = [
    ("upstream", "repository"),
    ("upstream", "commit"),
    ("upstream", "path"),
    ("upstream", "retrieved"),
    ("license", "spdx"),
    ("license", "file"),
    ("model", "calibration"),
    ("model", "mjcf"),
    ("model", "urdf"),
]
# Our own files sit in the folder but are not upstream copies, so they carry no hash.
OUR_FILES = {"PROVENANCE.toml", "NOTICE"}
# Text every copy of a license contains, to catch a LICENSE file that is some other license.
LICENSE_MARKERS = {"Apache-2.0": ("Apache License", "Version 2.0, January 2004")}
# Files the operating system adds on its own; not part of the vendored set.
IGNORED_FILES = {".DS_Store"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integrity_problems(root):
    """Everything wrong with the folder compared with its provenance record; empty when clean."""
    root = Path(root)
    record = tomllib.loads((root / "PROVENANCE.toml").read_text())
    problems = []
    for section, key in REQUIRED_FIELDS:
        if key not in record.get(section, {}):
            problems.append(f"provenance is missing {section}.{key}")
    commit = record.get("upstream", {}).get("commit")
    if commit is not None and not re.fullmatch(r"[0-9a-f]{40}", commit):
        problems.append("upstream.commit is not a 40-character commit hash")
    recorded = record.get("files", {})
    for name, digest in recorded.items():
        path = root / name
        if not path.is_file():
            problems.append(f"{name} is recorded but missing")
        elif sha256(path) != digest:
            problems.append(f"{name} differs from its recorded hash")
    license_name = record.get("license", {}).get("file")
    spdx = record.get("license", {}).get("spdx")
    if license_name and (root / license_name).is_file() and spdx in LICENSE_MARKERS:
        license_text = (root / license_name).read_text()
        if not all(marker in license_text for marker in LICENSE_MARKERS[spdx]):
            problems.append(f"{license_name} is not the {spdx} license text")
    notice = root / "NOTICE"
    if not notice.is_file():
        problems.append("NOTICE is missing")
    else:
        for section, key in [("upstream", "repository"), ("upstream", "commit"), ("license", "spdx")]:
            value = record.get(section, {}).get(key)
            if value is not None and str(value) not in notice.read_text():
                problems.append(f"NOTICE does not mention {section}.{key}")
    on_disk = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    for name in sorted(on_disk - set(recorded) - OUR_FILES - IGNORED_FILES):
        problems.append(f"{name} is vendored but not recorded")
    return problems
