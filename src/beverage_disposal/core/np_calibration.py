"""Neyman-Pearson calibration of the classifier's threshold (the umbrella algorithm).

The classifier may bin a bottle only when its score is above a threshold. The threshold is an
order statistic of the scores of calibration bottles that hold the threshold volume or more,
chosen so that, with probability at least 1 - delta over the calibration draw, at most a share
alpha of such bottles would score above it. Source: Tong, Feng and Li, "Neyman-Pearson
classification algorithms and NP receiver operating characteristics", Science Advances 2018.
"""

import math

import numpy as np


def min_samples(alpha: float, delta: float) -> int:
    """The fewest calibration scores for which any threshold can meet alpha with confidence.

    Even the largest of n scores fails with probability (1 - alpha)^n; that must be at most delta.
    """
    return math.ceil(math.log(delta) / math.log(1.0 - alpha))


def violation_probability(n: int, k: int, alpha: float) -> float:
    """The chance that the k-th smallest of n scores, as threshold, lets through more than alpha.

    It is P(Binomial(n, 1 - alpha) >= k), summed in log space so large n does not underflow.
    """
    if k <= 0:
        return 1.0
    j = np.arange(k, n + 1)
    log_terms = (
        math.lgamma(n + 1)
        - np.array([math.lgamma(v + 1) + math.lgamma(n - v + 1) for v in j])
        + j * math.log(1.0 - alpha)
        + (n - j) * math.log(alpha)
    )
    top = log_terms.max()
    return float(min(1.0, math.exp(top) * np.exp(log_terms - top).sum()))


def np_threshold(scores: np.ndarray, alpha: float, delta: float) -> float:
    """The smallest order statistic whose violation probability is at most delta."""
    n = len(scores)
    needed = min_samples(alpha, delta)
    if n < needed:
        raise ValueError(f"{n} calibration scores, but alpha and delta need at least {needed}")
    ordered = np.sort(scores)
    # The violation probability falls as k grows, so the first safe k is found by bisection.
    low, high = 1, n  # `high` is always safe: n >= min_samples
    while low < high:
        middle = (low + high) // 2
        if violation_probability(n, middle, alpha) <= delta:
            high = middle
        else:
            low = middle + 1
    return float(ordered[low - 1])


def calibrate_cells(
    scores: np.ndarray,
    sizes: np.ndarray,
    volumes: np.ndarray,
    bins: tuple[float, ...],
    alpha: float,
    delta: float,
) -> tuple[dict[str, float], list[tuple[str, float]]]:
    """A threshold per size: the strictest over its volume cells; and the cells that were empty.

    A cell is the bottles of one size whose volume lies from one bin edge up to the next (the
    last cell has no upper end). A cell with some rows but too few for alpha and delta raises.
    """
    if (volumes < bins[0]).any():
        raise ValueError(f"calibration rows under {bins[0]} mL are not rows the threshold guards")
    edges = [*bins, math.inf]
    taus: dict[str, float] = {}
    empty = []
    for size in dict.fromkeys(sizes.tolist()):
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            cell = scores[(sizes == size) & (volumes >= low) & (volumes < high)]
            if len(cell) == 0:
                empty.append((size, low))
                continue
            try:
                tau = np_threshold(cell, alpha, delta)
            except ValueError as error:
                raise ValueError(f"size {size}, cell from {low:g} mL: {error}") from error
            taus[size] = max(tau, taus.get(size, -math.inf))
    return taus, empty
