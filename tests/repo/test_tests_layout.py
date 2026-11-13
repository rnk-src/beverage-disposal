"""Only approved folders may sit directly in tests/, so the suite stays browsable as it grows."""

from support.gitenv import needs_git_work_tree, publishable_files
from support.layout import unexpected_top_level
from support.paths import ROOT

pytestmark = needs_git_work_tree

# Add a folder here when a new area of the project gets its own tests.
ALLOWED_IN_TESTS = [
    "conftest.py",
    "support",
    "repo",
    "environment",
    "tooling",
    "model",
    "gazebo",
    "kinematics",
    "control",
    "scene",
    "grasp",
    "scripted",
]


def test_only_approved_entries_sit_directly_in_the_tests_folder():
    published = [name for name in publishable_files(ROOT) if name.startswith("tests/")]
    inside_tests = [name.removeprefix("tests/") for name in published]
    assert inside_tests, "no published test files found, so nothing was checked"
    stray = unexpected_top_level(inside_tests, ALLOWED_IN_TESTS)
    assert not stray, f"move these into a topic folder, or approve a new folder here: {stray}"


def test_the_shared_root_is_the_repository_root_whatever_folder_a_test_sits_in():
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / "tests" / "conftest.py").is_file()
