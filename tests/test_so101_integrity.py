"""The 'untouched upstream copy' promise is enforced: every kind of damage is reported."""

import hashlib
import re
import shutil

import pytest
from so101 import SO101, integrity_problems

MESH = "assets/base_so101_v2.stl"


@pytest.fixture
def folder(tmp_path):
    """A private copy of the real vendored folder that a test may damage."""
    copy = tmp_path / "so101"
    shutil.copytree(SO101, copy)
    return copy


def drop_line(path, key):
    text = path.read_text()
    path.write_text(re.sub(rf"(?m)^{re.escape(key)} = .*\n", "", text))


def test_the_real_vendored_folder_has_no_problems():
    assert integrity_problems(SO101) == []


def test_the_copy_used_by_the_other_tests_starts_out_clean(folder):
    assert integrity_problems(folder) == []


def test_one_flipped_byte_in_a_mesh_is_reported(folder):
    data = bytearray((folder / MESH).read_bytes())
    data[1000] ^= 0x01
    (folder / MESH).write_bytes(bytes(data))
    assert integrity_problems(folder) == [f"{MESH} differs from its recorded hash"]


def test_a_deleted_mesh_is_reported(folder):
    (folder / MESH).unlink()
    assert integrity_problems(folder) == [f"{MESH} is recorded but missing"]


def test_a_missing_license_file_is_reported(folder):
    (folder / "LICENSE").unlink()
    assert integrity_problems(folder) == ["LICENSE is recorded but missing"]


def test_an_extra_file_that_provenance_does_not_record_is_reported(folder):
    (folder / "assets" / "extra.stl").write_bytes(b"x")
    assert integrity_problems(folder) == ["assets/extra.stl is vendored but not recorded"]


def test_a_file_the_finder_may_drop_in_the_folder_is_not_a_problem(folder):
    (folder / ".DS_Store").write_bytes(b"x")
    assert integrity_problems(folder) == []


def test_an_altered_recorded_hash_is_reported(folder):
    provenance = folder / "PROVENANCE.toml"
    text = provenance.read_text()
    real = re.search(rf'"{MESH}" = "([0-9a-f]{{64}})"', text).group(1)
    provenance.write_text(text.replace(real, "0" * 64))
    assert integrity_problems(folder) == [f"{MESH} differs from its recorded hash"]


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("upstream", "repository"),
        ("upstream", "commit"),
        ("upstream", "path"),
        ("upstream", "retrieved"),
        ("license", "spdx"),
        ("license", "file"),
        ("model", "calibration"),
        ("model", "mjcf"),
        ("model", "urdf"),
    ],
)
def test_a_missing_provenance_field_is_reported(folder, section, key):
    drop_line(folder / "PROVENANCE.toml", key)
    assert integrity_problems(folder) == [f"provenance is missing {section}.{key}"]


def test_a_commit_that_is_a_branch_name_and_not_a_hash_is_reported(folder):
    provenance = folder / "PROVENANCE.toml"
    text = provenance.read_text()
    commit = re.search(r'commit = "([0-9a-f]{40})"', text).group(1)
    provenance.write_text(text.replace(commit, "main"))
    notice = folder / "NOTICE"
    notice.write_text(notice.read_text().replace(commit, "main"))
    assert integrity_problems(folder) == ["upstream.commit is not a 40-character commit hash"]


def test_a_missing_notice_is_reported(folder):
    (folder / "NOTICE").unlink()
    assert integrity_problems(folder) == ["NOTICE is missing"]


@pytest.mark.parametrize(
    ("section", "field", "other_value"),
    [
        ("upstream", "repository", "https://example.org/other/project"),
        ("upstream", "commit", "0" * 40),
        ("license", "spdx", "GPL-3.0"),
    ],
)
def test_a_notice_that_no_longer_matches_the_provenance_is_reported(folder, section, field, other_value):
    provenance = folder / "PROVENANCE.toml"
    text = provenance.read_text()
    old = re.search(rf'{field} = "([^"]+)"', text).group(1)
    provenance.write_text(text.replace(old, other_value))
    assert integrity_problems(folder) == [f"NOTICE does not mention {section}.{field}"]


def test_a_license_file_that_is_not_the_recorded_license_is_reported(folder):
    (folder / "LICENSE").write_text("MIT License\n\nPermission is hereby granted, free of charge.\n")
    digest = hashlib.sha256((folder / "LICENSE").read_bytes()).hexdigest()
    provenance = folder / "PROVENANCE.toml"
    text = provenance.read_text()
    old = re.search(r'"LICENSE" = "([0-9a-f]{64})"', text).group(1)
    provenance.write_text(text.replace(old, digest))
    assert integrity_problems(folder) == ["LICENSE is not the Apache-2.0 license text"]
