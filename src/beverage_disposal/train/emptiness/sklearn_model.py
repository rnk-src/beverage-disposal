"""A scikit-learn classifier behind the core's `EmptinessModel` interface."""

import numpy as np

EMPTY_CLASS = 1  # the label of "holds under the threshold" in the fitted estimator


class SklearnModel:
    def __init__(self, estimator):
        self.estimator = estimator

    def score(self, x: np.ndarray) -> float:
        return float(self.scores(x.reshape(1, -1))[0])

    def scores(self, rows: np.ndarray) -> np.ndarray:
        """The score of every row at once: the estimated chance each holds under the threshold."""
        column = list(self.estimator.classes_).index(EMPTY_CLASS)
        return self.estimator.predict_proba(rows)[:, column]
