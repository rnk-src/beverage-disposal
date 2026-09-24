"""What the published ignore file does, checked through git itself."""

import pytest
from support.gitenv import ignored_by_published_file, is_ignored, needs_git_work_tree
from support.paths import ROOT

pytestmark = needs_git_work_tree

# Hand-picked: one path per kind of thing that must never be committed.
IGNORED = [
    ".venv/bin/python",
    "src/beverage_disposal/__pycache__/x.cpython-312.pyc",
    ".pytest_cache/README.md",
    "runs/exp1/events.out.tfevents",
    ".DS_Store",
    ".idea/workspace.xml",
    ".vscode/settings.json",
    ".claude/settings.local.json",
]

# Real project files that must stay visible to git, including a source package that happens to
# share the name of the training-output directory.
NOT_IGNORED = [
    "src/beverage_disposal/__init__.py",
    "src/beverage_disposal/runs/__init__.py",
    "tests/test_something.py",
    "assets/so101/model.xml",
    "configs/task.yaml",
    "scripts/viewer.py",
    "README.md",
]


@pytest.mark.parametrize("path", IGNORED)
def test_generated_and_per_machine_paths_are_ignored(path):
    assert ignored_by_published_file(ROOT, path)


@pytest.mark.parametrize("path", NOT_IGNORED)
def test_real_source_is_not_ignored(path):
    assert not is_ignored(ROOT, path)
