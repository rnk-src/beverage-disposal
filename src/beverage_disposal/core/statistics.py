"""Confidence intervals for counts, so a result can be stated as 'k of n, bound X %'."""

import math


def wilson_interval(failures: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for a rate seen as `failures` in `trials` (z = 1.96 is 95 %)."""
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if not 0 <= failures <= trials:
        raise ValueError(f"failures must be between 0 and trials, got {failures} of {trials}")
    rate = failures / trials
    denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    # The interval's ends are exactly 0 and 1 at the extremes; rounding would leave them a hair off.
    low = 0.0 if failures == 0 else max(0.0, centre - half)
    high = 1.0 if failures == trials else min(1.0, centre + half)
    return low, high


def wilson_upper(failures: int, trials: int, z: float = 1.96) -> float:
    """The upper end of the Wilson interval: the rate the data cannot rule out."""
    return wilson_interval(failures, trials, z)[1]
