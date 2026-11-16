"""Neyman-Pearson calibration: the sample size, the violation probability and the threshold."""

import math

import numpy as np
import pytest

from beverage_disposal.core.np_calibration import (
    calibrate_cells,
    min_samples,
    np_threshold,
    violation_probability,
)
from beverage_disposal.core.statistics import wilson_upper


def test_the_fewest_samples_are_the_first_n_with_all_n_below_alpha_unlikely_enough():
    # (1 - alpha)^n <= delta: 0.99^298 = 0.0500..., above; 0.99^299 below.
    assert 0.99**298 > 0.05 > 0.99**299
    assert min_samples(0.01, 0.05) == 299
    assert 0.995**597 > 0.05 > 0.995**598
    assert min_samples(0.005, 0.05) == 598


@pytest.mark.parametrize("k", [1, 10, 17, 19, 20])
def test_the_violation_probability_is_a_binomial_tail(k):
    n, alpha = 20, 0.1
    direct = sum(math.comb(n, j) * 0.9**j * 0.1 ** (n - j) for j in range(k, n + 1))
    assert violation_probability(n, k, alpha) == pytest.approx(direct, rel=1e-10)


def test_the_violation_probability_at_the_ends():
    assert violation_probability(598, 598, 0.005) == pytest.approx(0.995**598)
    assert violation_probability(50, 0, 0.1) == pytest.approx(1.0)
    assert violation_probability(5000, 5000, 0.005) == pytest.approx(0.995**5000, rel=1e-9)


def test_with_just_enough_samples_the_threshold_is_the_largest_score():
    scores = np.random.default_rng(0).uniform(size=598)
    assert np_threshold(scores, 0.005, 0.05) == scores.max()


def test_with_more_samples_the_threshold_is_the_smallest_order_statistic_that_is_safe():
    scores = np.random.default_rng(1).uniform(size=3000)
    ordered = np.sort(scores)
    safe = [k for k in range(1, 3001) if violation_probability(3000, k, 0.005) <= 0.05]
    assert np_threshold(scores, 0.005, 0.05) == ordered[safe[0] - 1]
    # It sits above the plain 99.5 % quantile: the price of the confidence.
    assert safe[0] > math.ceil(0.995 * 3000)
    assert violation_probability(3000, safe[0] - 1, 0.005) > 0.05


def test_too_few_samples_are_refused():
    with pytest.raises(ValueError, match="598"):
        np_threshold(np.zeros(597), 0.005, 0.05)


def test_a_constant_score_gives_that_constant_so_a_strict_comparison_never_passes():
    tau = np_threshold(np.full(700, 0.83), 0.005, 0.05)
    assert tau == 0.83 and not 0.83 > tau


@pytest.mark.medium
def test_over_many_calibrations_the_error_exceeds_alpha_no_more_often_than_delta():
    """Uniform scores: a threshold t lets through 1 - t, so a violation is t < 1 - alpha."""
    rng = np.random.default_rng(2)
    alpha, delta, runs = 0.05, 0.1, 2000
    violations = sum(
        np_threshold(rng.uniform(size=200), alpha, delta) < 1 - alpha for _ in range(runs)
    )
    assert violations / runs <= delta
    # And it is not vacuous: a plain quantile would be violated about half the time.
    plain = sum(np.quantile(rng.uniform(size=200), 1 - alpha) < 1 - alpha for _ in range(runs))
    assert plain / runs > 0.3
    assert wilson_upper(violations, runs) < 0.15


def cell_scores():
    """Two sizes; size a has two cells with different score levels, size b one cell."""
    rng = np.random.default_rng(3)
    scores = np.concatenate(
        [rng.uniform(0.0, 0.5, 700), rng.uniform(0.0, 0.9, 700), rng.uniform(0.0, 0.2, 700)]
    )
    sizes = np.array(["a"] * 1400 + ["b"] * 700)
    volumes = np.concatenate([np.full(700, 30.5), np.full(700, 40.0), np.full(700, 60.0)])
    return scores, sizes, volumes


def test_a_sizes_threshold_is_the_strictest_of_its_cells():
    scores, sizes, volumes = cell_scores()
    taus, empty = calibrate_cells(scores, sizes, volumes, (30.0, 31.0, 35.0, 50.0), 0.005, 0.05)
    first = np_threshold(scores[:700], 0.005, 0.05)
    second = np_threshold(scores[700:1400], 0.005, 0.05)
    assert second > first
    assert taus == {"a": second, "b": np_threshold(scores[1400:], 0.005, 0.05)}


def test_empty_cells_are_listed_and_a_volume_on_an_edge_belongs_to_the_cell_above():
    scores, sizes, volumes = cell_scores()
    taus, empty = calibrate_cells(scores, sizes, volumes, (30.0, 31.0, 35.0, 50.0), 0.005, 0.05)
    assert set(empty) == {("a", 31.0), ("a", 50.0), ("b", 30.0), ("b", 31.0), ("b", 35.0)}
    on_edge = np.where(volumes == 30.5, 31.0, volumes)
    _, moved = calibrate_cells(scores, sizes, on_edge, (30.0, 31.0, 35.0, 50.0), 0.005, 0.05)
    assert ("a", 30.0) in moved and ("a", 31.0) not in moved


def test_a_cell_with_some_rows_but_too_few_is_refused_by_name():
    scores, sizes, volumes = cell_scores()
    with pytest.raises(ValueError, match="b.*60|b.*50"):
        calibrate_cells(scores[:-200], sizes[:-200], volumes[:-200], (30.0, 50.0), 0.005, 0.05)


def test_rows_under_the_first_edge_are_not_calibration_rows():
    scores, sizes, volumes = cell_scores()
    with pytest.raises(ValueError, match="under"):
        calibrate_cells(scores, sizes, volumes - 10.0, (30.0, 50.0), 0.005, 0.05)
