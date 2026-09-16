import pytest

from beverage_disposal_bringup.disposal_decision import decide_disposal


def test_empty_fullness_decides_bin():
    assert decide_disposal('empty') == 'bin'


def test_full_fullness_decides_set_aside():
    assert decide_disposal('full') == 'set_aside'


@pytest.mark.parametrize('invalid_fullness', [None, 'unknown', '', 0])
def test_invalid_fullness_raises(invalid_fullness):
    with pytest.raises(ValueError):
        decide_disposal(invalid_fullness)
