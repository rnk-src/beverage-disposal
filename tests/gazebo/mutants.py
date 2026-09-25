"""Runs the Gazebo tests against a damaged copy of the repository, to prove they can fail.

The copy holds tests/, docker/ and pyproject.toml, so the tests' own repository root points at the
copy and the real files are never touched. Each damaged image gets its own tag and is removed after.
"""

import os
import shutil
import subprocess
import sys
import uuid

from support.paths import ROOT


def edit(folder, name, old, new, everywhere=False):
    path = folder / name
    text = path.read_text()
    assert old in text, f"{old!r} not found in {name}: the fault would not change anything"
    path.write_text(text.replace(old, new, -1 if everywhere else 1))


def run_on_damaged_copy(tmp_path, fault, selection, env=None):
    """Copy the repository pieces, apply `fault(copy_root)`, run `selection`, return (result, ids).

    `ids` maps outcome ("FAILED", "SKIPPED", "PASSED") to the test names reported with that outcome.
    """
    copy = tmp_path / "copy"
    copy.mkdir()
    shutil.copytree(
        ROOT / "tests", copy / "tests", ignore=shutil.ignore_patterns("private", "__pycache__")
    )
    shutil.copytree(ROOT / "docker", copy / "docker")
    shutil.copy(ROOT / "pyproject.toml", copy / "pyproject.toml")
    fault(copy)
    tag = f"beverage-disposal-gazebo:fault-{uuid.uuid4().hex[:8]}"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", selection, "-q", "--tb=no", "-rfsp"]
            + ["-p", "no:cacheprovider", "-o", "addopts="],
            cwd=copy,
            env={
                **os.environ,
                "GAZEBO_IMAGE_TAG": tag,
                "PYTHONDONTWRITEBYTECODE": "1",
                **(env or {}),
            },
            capture_output=True,
            text=True,
        )
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
    return result, outcomes(result.stdout)


def outcomes(stdout):
    found = {"FAILED": [], "SKIPPED": [], "PASSED": []}
    for line in stdout.splitlines():
        word, _, rest = line.partition(" ")
        if word in found and "::" in rest:
            found[word].append(rest.split(" - ")[0].split("::", 1)[1])
    return found
