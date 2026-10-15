"""Published docs must not carry anything that identifies a person or a machine."""

import pytest
from support.gitenv import needs_git_work_tree, publishable_files
from support.paths import ROOT
from support.scanning import (
    SCANNER_SAMPLE_DATA,
    find_leaks_in_files,
    find_private_looking_text,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see /Users/alice/project/notes.md", ["/Users/alice"]),
        ("cd /home/bob/.config and look", ["/home/bob"]),
        (r"open C:\Users\carol\code now", [r"C:\Users\carol"]),
    ],
)
def test_scanner_flags_absolute_home_paths(text, expected):
    assert find_private_looking_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "write to alice@example.com please",
        "mail alice@example.com.",
        "ask alice@example.com: thanks",
    ],
)
def test_scanner_flags_email_addresses(text):
    assert find_private_looking_text(text) == ["alice@example.com"]


@pytest.mark.parametrize(
    "text",
    [
        "mark slow tests with `@pytest.mark.slow`",
        "the skill lives at https://github.com/obra/superpowers",
        "caches sit under ~/.cache and binaries under /usr/local/bin",
        "clone it with git clone git@github.com:owner/repo.git",
        "or use the remote ssh://git@github.com/owner/repo.git",
    ],
)
def test_scanner_leaves_ordinary_text_alone(text):
    assert find_private_looking_text(text) == []


@needs_git_work_tree
def test_published_text_files_carry_no_home_paths_or_emails():
    files = publishable_files(ROOT)
    assert files, "no published files found, so nothing was scanned"
    assert find_leaks_in_files(ROOT, files, exempt=SCANNER_SAMPLE_DATA) == {}


def test_any_text_file_is_scanned_not_only_docs(tmp_path):
    (tmp_path / "config.yaml").write_text("data_dir: /Users/alice/data\n")
    assert find_leaks_in_files(tmp_path, ["config.yaml"]) == {"config.yaml": ["/Users/alice"]}


def test_binary_files_are_skipped(tmp_path):
    (tmp_path / "mesh.stl").write_bytes(b"solid\x00\x01/Users/alice/x\x00")
    assert find_leaks_in_files(tmp_path, ["mesh.stl"]) == {}


def test_an_exempt_file_is_skipped_but_its_neighbours_are_not(tmp_path):
    (tmp_path / "samples.py").write_text("PATH = '/Users/alice/x'\n")
    (tmp_path / "other.py").write_text("PATH = '/Users/alice/x'\n")
    found = find_leaks_in_files(tmp_path, ["samples.py", "other.py"], exempt=["samples.py"])
    assert found == {"other.py": ["/Users/alice"]}


def test_a_listed_file_that_is_missing_from_disk_is_skipped_not_a_crash(tmp_path):
    # git still lists a tracked file that was deleted but not staged, so the file is missing.
    (tmp_path / "there.py").write_text("path = '/Users/alice/data'\n")
    assert find_leaks_in_files(tmp_path, ["gone.py", "there.py"]) == {"there.py": ["/Users/alice"]}
