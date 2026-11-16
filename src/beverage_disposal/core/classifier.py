"""The emptiness classifier as the policy sees it: a score, a range check and a bundle.

No learning library is imported here. The model is anything with a `score`; the training side
fits one and the policy only asks it.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from beverage_disposal.core.features import MAY_BE_MISSING


class EmptinessModel(Protocol):
    def score(self, x: np.ndarray) -> float:
        """How sure the model is that the bottle holds under the threshold: higher is emptier."""
        ...


@dataclass(frozen=True)
class FeatureRanges:
    """The span of each feature in the training rows: outside it the model is not trusted."""

    lo: tuple[float, ...]
    hi: tuple[float, ...]
    may_be_missing: tuple[bool, ...]  # where NaN is an ordinary value

    @classmethod
    def from_rows(cls, rows: np.ndarray, names: tuple[str, ...]) -> "FeatureRanges":
        """The smallest and largest value of every feature over `rows`, missing values aside."""
        missing_ok = tuple(name in MAY_BE_MISSING for name in names)
        for name, allowed, column in zip(names, missing_ok, rows.T, strict=True):
            if not allowed and np.isnan(column).any():
                raise ValueError(f"{name} is missing in some rows and must never be")
        return cls(
            tuple(float(v) for v in np.nanmin(rows, axis=0)),
            tuple(float(v) for v in np.nanmax(rows, axis=0)),
            missing_ok,
        )

    def in_distribution(self, x: np.ndarray) -> bool:
        """Whether every feature lies in its range; a missing value passes only where allowed."""
        if x.shape != (len(self.lo),):
            return False
        missing = np.isnan(x)
        inside = (x >= np.array(self.lo)) & (x <= np.array(self.hi))
        return bool(np.all(np.where(missing, np.array(self.may_be_missing), inside)))


@dataclass(frozen=True)
class ModelBundle:
    """A fitted model with everything the policy needs to use it, and what it was made from."""

    model: EmptinessModel
    tau_by_size: dict[str, float]  # the score a bottle of a size must beat to be binned
    ranges: FeatureRanges
    feature_names: tuple[str, ...]
    model_sha256: str
    meta: dict
