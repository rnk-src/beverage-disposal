"""The Python version the project allows and pins."""

import tomllib

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from support.paths import ROOT

REQUIRES_PYTHON = SpecifierSet(
    tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
)


def test_the_pinned_python_satisfies_the_allowed_range():
    pinned = Version((ROOT / ".python-version").read_text().strip())
    assert pinned in REQUIRES_PYTHON


@pytest.mark.parametrize(
    ("version", "allowed"),
    [
        ("3.11.9", False),
        ("3.12.0", True),
        ("3.12.13", True),
        ("3.13.0", False),
    ],
)
def test_only_python_3_12_is_allowed(version, allowed):
    assert (Version(version) in REQUIRES_PYTHON) is allowed
