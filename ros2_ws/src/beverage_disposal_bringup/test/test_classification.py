from beverage_disposal_bringup.classification import (
    EMPTY_CAN_MASS_KG,
    FULL_CAN_MASS_KG,
    FULLNESS_MASS_THRESHOLD_KG,
    classify_fullness,
)


def test_low_mass_reads_as_empty():
    assert classify_fullness(FULLNESS_MASS_THRESHOLD_KG - 0.1) == 'empty'


def test_high_mass_reads_as_full():
    assert classify_fullness(FULLNESS_MASS_THRESHOLD_KG + 0.1) == 'full'


def test_mass_at_threshold_reads_as_full():
    assert classify_fullness(FULLNESS_MASS_THRESHOLD_KG) == 'full'


def test_real_empty_can_mass_reads_as_empty():
    assert classify_fullness(EMPTY_CAN_MASS_KG) == 'empty'


def test_real_full_can_mass_reads_as_full():
    assert classify_fullness(FULL_CAN_MASS_KG) == 'full'
