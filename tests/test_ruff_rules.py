"""What the configured lint rules catch, checked on tiny one-defect snippets."""

import pytest
from ruffrun import lint_codes

# 100 characters wide, and 101: the line-length limit is 100.
LINE_100 = 'x = "' + "a" * 94 + '"\n'
LINE_101 = 'x = "' + "a" * 95 + '"\n'

CLEAN = '''"""Adds up some numbers."""

import os


def total(values: list[int]) -> int:
    return sum(values) + len(os.sep)
'''


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import os\n", ["F401"]),
        ("print(undefined_name)\n", ["F821"]),
        ("x = 1\nif x == None:\n    pass\n", ["E711"]),
        (LINE_101, ["E501"]),
        ("import sys\nimport os\n\nprint(os, sys)\n", ["I001"]),
        ("class Plain(object):\n    pass\n", ["UP004"]),
        ("def add(item, seen=[]):\n    return seen + [item]\n", ["B006"]),
    ],
    ids=["F401", "F821", "E711", "E501", "I001", "UP004", "B006"],
)
def test_a_one_defect_snippet_is_reported_with_exactly_that_rule(tmp_path, source, expected):
    assert lint_codes(tmp_path, source) == expected


def test_a_line_exactly_at_the_limit_is_fine(tmp_path):
    assert len(LINE_100.rstrip("\n")) == 100
    assert lint_codes(tmp_path, LINE_100) == []


def test_clean_code_is_not_reported(tmp_path):
    assert lint_codes(tmp_path, CLEAN) == []
