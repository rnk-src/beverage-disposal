"""Git helpers shared by the public and the private tests. Generic on purpose: no private names."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# Git reads nothing outside the repo, so a rule in someone's global ignore cannot change the answer.
# Git also sets GIT_INDEX_FILE and friends for a hook that runs during a commit; inherited here they
# would point these tests at the real repository instead of the one they were given.
ISOLATED_GIT = {
    **{name: value for name, value in os.environ.items() if not name.startswith("GIT_")},
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "XDG_CONFIG_HOME": os.devnull,
}


def is_ignored(repo, path):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=repo, env=ISOLATED_GIT)
    assert result.returncode in (0, 1), f"git check-ignore failed with {result.returncode}"
    return result.returncode == 0


def ignore_source(repo, path):
    """The file that supplies the ignore rule for `path`, or None if git does not ignore it."""
    result = subprocess.run(
        ["git", "check-ignore", "-v", path],
        cwd=repo,
        env=ISOLATED_GIT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 1:
        return None
    assert result.returncode == 0, result.stderr
    return re.match(r"(.*?):\d+:", result.stdout).group(1)


def ignored_by_published_file(repo, path):
    """True only if the rule comes from the published ignore file, not a local exclude rule."""
    return ignore_source(repo, path) == ".gitignore"


def publishable_files(repo):
    """Everything git would publish: tracked files plus untracked files that are not ignored."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=repo,
        env=ISOLATED_GIT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def is_git_work_tree(path):
    if shutil.which("git") is None:
        return False
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=path,
        env=ISOLATED_GIT,
        capture_output=True,
    )
    return result.returncode == 0


# Tests that ask git questions about this repo are skipped on an exported copy with no .git.
needs_git_work_tree = pytest.mark.skipif(
    not is_git_work_tree(Path(__file__).resolve().parents[1]),
    reason="not inside a git work tree",
)
