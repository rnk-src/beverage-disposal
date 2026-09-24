"""What pyproject.toml declares: which packages, and how tightly each one is bounded."""

import re
import tomllib

import pytest
from support.paths import ROOT

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())

RUNTIME = {"mujoco", "numpy", "gymnasium"}
# packaging is what test_python_pin.py uses to compare versions; declared so it is not borrowed.
DEV = {"pytest", "hypothesis", "packaging", "ruff", "pre-commit", "pillow"}


def package_name(spec):
    return re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0]


def all_specs():
    return PYPROJECT["project"]["dependencies"] + PYPROJECT["dependency-groups"]["dev"]


def test_runtime_dependencies_are_exactly_the_decided_ones():
    declared = {package_name(spec) for spec in PYPROJECT["project"]["dependencies"]}
    assert declared == RUNTIME


def test_dev_dependencies_are_exactly_the_decided_ones():
    declared = {package_name(spec) for spec in PYPROJECT["dependency-groups"]["dev"]}
    assert declared == DEV


@pytest.mark.parametrize("spec", all_specs())
def test_every_range_has_a_lower_and_an_upper_bound_and_is_not_an_exact_pin(spec):
    assert ">=" in spec, f"{spec}: no lower bound"
    assert re.search(r"<\d", spec), f"{spec}: no upper bound"
    assert "==" not in spec, f"{spec}: exact pins belong in uv.lock"
