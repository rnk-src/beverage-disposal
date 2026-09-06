from beverage_disposal_bringup.classification import (
    FULLNESS_EFFORT_THRESHOLD,
    classify_fullness,
)


def test_low_effort_reads_as_empty():
    assert classify_fullness(FULLNESS_EFFORT_THRESHOLD - 0.1) == 'empty'


def test_high_effort_reads_as_full():
    assert classify_fullness(FULLNESS_EFFORT_THRESHOLD + 0.1) == 'full'


def test_effort_at_threshold_reads_as_full():
    assert classify_fullness(FULLNESS_EFFORT_THRESHOLD) == 'full'
