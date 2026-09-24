"""What the configured formatter does, checked on tiny snippets."""

import pytest
from ruffrun import formatted


def call_of_width(columns):
    """`value = call("aaa...")` exactly `columns` wide: 14 chars before the a's, 2 after."""
    line = 'value = call("' + "a" * (columns - 16) + '")\n'
    assert len(line.rstrip("\n")) == columns
    return line


WRAPPED_101 = 'value = call(\n    "' + "a" * 85 + '"\n)\n'


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x = { 'a':1 }\n", 'x = {"a": 1}\n'),
        (call_of_width(100), call_of_width(100)),
        (call_of_width(101), WRAPPED_101),
    ],
    ids=["quotes-and-spacing", "100-columns-stays", "101-columns-wraps"],
)
def test_the_formatter_produces_the_expected_text(tmp_path, source, expected):
    assert formatted(tmp_path, source) == expected
