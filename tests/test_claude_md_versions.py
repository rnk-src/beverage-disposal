"""CLAUDE.md states the locked versions; they must not drift from uv.lock."""

from pathlib import Path

import pytest
import lockversions as versions

ROOT = Path(__file__).resolve().parents[1]

LOCK = """
[[package]]
name = "mujoco"
version = "3.14.0"

[[package]]
name = "numpy"
version = "2.5.3"
"""


def claude_md(sentence):
    return f"# T\n\n- **Tooling:** Python 3.12 pinned, uv 0.11 used.\n  {sentence}\n- **Next:** numpy 9.9.9 is unrelated.\n"


def test_versions_are_read_from_the_locked_versions_sentence_only():
    text = claude_md("Versions locked 2026-09-24: mujoco 3.14.0, numpy 2.5.3.")
    assert versions.claude_md_versions(text) == {"mujoco": "3.14.0", "numpy": "2.5.3"}


def test_a_sentence_that_wraps_over_lines_is_read_whole():
    text = claude_md("Versions locked 2026-09-24: mujoco 3.14.0,\n  numpy 2.5.3.")
    assert versions.claude_md_versions(text) == {"mujoco": "3.14.0", "numpy": "2.5.3"}


def test_no_locked_versions_sentence_is_an_error_not_a_silent_pass():
    with pytest.raises(ValueError, match="Versions locked"):
        versions.claude_md_versions("# T\n\nnothing about versions here\n")


def test_the_lockfile_versions_are_read_by_package_name():
    assert versions.locked_versions(LOCK) == {"mujoco": "3.14.0", "numpy": "2.5.3"}


def test_matching_versions_give_no_drift():
    assert versions.drift({"mujoco": "3.14.0"}, versions.locked_versions(LOCK)) == []


def test_a_different_version_is_reported_with_both_values():
    found = versions.drift({"numpy": "2.5.2"}, versions.locked_versions(LOCK))
    assert found == ["numpy: stated 2.5.2, uv.lock has 2.5.3"]


def test_a_package_that_is_not_locked_is_reported():
    found = versions.drift({"jax": "0.4.1"}, versions.locked_versions(LOCK))
    assert found == ["jax: stated 0.4.1, not in uv.lock"]


def test_the_versions_claude_md_states_match_uv_lock():
    stated = versions.claude_md_versions((ROOT / "CLAUDE.md").read_text())
    drift = versions.drift(stated, versions.locked_versions((ROOT / "uv.lock").read_text()))
    assert not drift, "; ".join(drift)
