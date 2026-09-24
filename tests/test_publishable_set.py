"""Only known top-level entries may be publishable, so a stray private folder cannot slip out."""

from pathlib import Path

import pytest
from gitenv import needs_git_work_tree, publishable_files
from layout import explain_unexpected, unexpected_top_level

ALLOWED = ["README.md", "src", "tests"]


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (["README.md", "src/pkg/__init__.py", "tests/test_a.py"], []),
        (["README.md", "notes/plan.md"], ["notes"]),
        (["README.md", "todo.txt", "notes/a.md", "notes/b.md"], ["notes", "todo.txt"]),
        (["src/pkg/README.md"], []),
        ([], []),
    ],
)
def test_entries_outside_the_allowlist_are_reported_once_each_and_sorted(files, expected):
    assert unexpected_top_level(files, ALLOWED) == expected


def _stray_private_notes(repo):
    (repo / "README.md").write_text("hello")
    (repo / "notes").mkdir()
    (repo / "notes" / "plan.md").write_text("private thoughts")


def test_a_fresh_clone_without_local_exclude_rules_trips_the_guard(repo):
    _stray_private_notes(repo)
    assert unexpected_top_level(publishable_files(repo), ALLOWED) == ["notes"]


def test_the_guard_is_quiet_when_the_local_exclude_rule_exists(repo):
    _stray_private_notes(repo)
    (repo / ".git" / "info" / "exclude").write_text("/notes\n")
    assert unexpected_top_level(publishable_files(repo), ALLOWED) == []


def test_the_message_names_the_entry_and_both_remedies():
    message = explain_unexpected(["notes"])
    assert "notes" in message
    assert "allowlist" in message
    assert "exclude" in message


ROOT = Path(__file__).resolve().parents[1]

# Top-level entries that belong to the project. Add a new one here when the project grows.
PROJECT_TOP_LEVEL = [
    ".gitignore",
    ".python-version",
    "CLAUDE.md",
    "LICENSE",
    "README.md",
    "assets",
    "configs",
    "pyproject.toml",
    "scripts",
    "src",
    "tests",
    "uv.lock",
]


@needs_git_work_tree
def test_git_would_only_publish_known_top_level_entries():
    files = publishable_files(ROOT)
    assert files, "no published files found, so nothing was checked"
    stray = unexpected_top_level(files, PROJECT_TOP_LEVEL)
    assert not stray, explain_unexpected(stray)
