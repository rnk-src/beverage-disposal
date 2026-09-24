"""The scanner for home paths and emails still reads the vendored files."""

import shutil
from pathlib import Path

import pytest
from gitenv import needs_git_work_tree, publishable_files
from scanning import find_leaks_in_files
from so101 import SO101, URDF
from test_docs_stay_generic import SCANNER_SAMPLE_DATA

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("planted", "expected"),
    [
        ("/Users/alice/robots", "/Users/alice"),
        ("alice@example.com", "alice@example.com"),
    ],
)
def test_the_scanner_reports_private_looking_text_inside_a_vendored_file(
    tmp_path, planted, expected
):
    copy = tmp_path / "so101"
    shutil.copytree(SO101, copy)
    with (copy / URDF).open("a") as handle:
        handle.write(f"<!-- {planted} -->\n")
    assert find_leaks_in_files(copy, [URDF]) == {URDF: [expected]}


@needs_git_work_tree
def test_the_vendored_text_files_are_scanned_and_not_exempted():
    published = set(publishable_files(ROOT))
    vendored_text = {
        f"assets/so101/{name}"
        for name in (
            "so101_new_calib.xml",
            "so101_new_calib.urdf",
            "README.md",
            "LICENSE",
            "NOTICE",
            "PROVENANCE.toml",
        )
    }
    assert vendored_text <= published
    assert not [name for name in SCANNER_SAMPLE_DATA if name.startswith("assets/")]
