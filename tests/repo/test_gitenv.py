"""The git helpers the other tests rely on, checked against throwaway repos."""

import importlib
import subprocess

import pytest
from support import gitenv
from support.gitenv import ignored_by_published_file, is_git_work_tree


def test_a_rule_in_the_published_ignore_file_counts(repo):
    (repo / ".gitignore").write_text("draft.md\n")
    assert ignored_by_published_file(repo, "draft.md")


def test_a_rule_only_in_the_local_exclude_file_does_not_count(repo):
    (repo / ".git" / "info" / "exclude").write_text("draft.md\n")
    assert not ignored_by_published_file(repo, "draft.md")


def test_a_path_with_no_rule_does_not_count(repo):
    assert not ignored_by_published_file(repo, "draft.md")


def test_a_git_repo_is_a_work_tree(repo):
    assert is_git_work_tree(repo)


def test_a_plain_folder_is_not_a_work_tree(tmp_path):
    assert not is_git_work_tree(tmp_path)


def test_a_machine_without_git_is_not_a_work_tree(repo, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert not is_git_work_tree(repo)


@pytest.mark.parametrize("variable", ["GIT_DIR", "GIT_INDEX_FILE"])
def test_git_variables_set_for_hooks_do_not_reach_the_test_repos(
    variable, repo, tmp_path_factory, monkeypatch
):
    # git sets these for a hook that runs during a commit; the value here points at another repo.
    other = tmp_path_factory.mktemp("other")
    subprocess.run(["git", "init", "-q"], cwd=other, check=True)
    (other / "theirs.txt").write_text("staged in the other repo")
    subprocess.run(["git", "add", "theirs.txt"], cwd=other, check=True)
    hostile = other / ".git" if variable == "GIT_DIR" else other / ".git" / "index"
    monkeypatch.setenv(variable, str(hostile))
    reloaded = importlib.reload(gitenv)
    try:
        (repo / "mine.txt").write_text("x")
        assert reloaded.publishable_files(repo) == ["mine.txt"]
    finally:
        monkeypatch.undo()
        importlib.reload(gitenv)
