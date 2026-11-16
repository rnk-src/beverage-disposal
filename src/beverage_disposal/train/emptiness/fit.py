"""Fits the emptiness classifier on simulated bottles."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from beverage_disposal.train.emptiness.dataset import Dataset
from beverage_disposal.train.emptiness.sklearn_model import SklearnModel

# The zones the classifier is ever asked about; a surely-over bottle never reaches it.
FIT_ZONES = ("surely_under", "ambiguous")


@dataclass(frozen=True)
class FitConfig:
    max_iter: int = 200
    learning_rate: float = 0.1
    max_leaf_nodes: int = 31
    random_state: int = 0


def fit(train: Dataset, config: FitConfig) -> SklearnModel:
    """Gradient-boosted trees on the rows the classifier can be asked about.

    Histogram boosting takes the missing level as it is, and with a fixed `random_state` and no
    early stopping the same rows give the same model.
    """
    rows = np.isin(train.zone, FIT_ZONES)
    if not rows.any():
        raise ValueError("no row in a zone the classifier is asked about")
    estimator = HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.max_leaf_nodes,
        early_stopping=False,
        random_state=config.random_state,
    )
    estimator.fit(train.X[rows], train.y[rows].astype(int))
    return SklearnModel(estimator)
