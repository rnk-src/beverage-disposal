"""Saves and loads a fitted model with its thresholds, ranges and provenance.

The model is pickled, and unpickling runs code: load only bundles this repository wrote. The
checksum catches a damaged or swapped file, not a hostile one.
"""

import dataclasses
import hashlib
import json
import pickle
from pathlib import Path

import sklearn

from beverage_disposal.core.classifier import FeatureRanges, ModelBundle
from beverage_disposal.train.emptiness.sklearn_model import SklearnModel

MODEL_FILE = "model.pkl"
BUNDLE_FILE = "bundle.json"


def save_bundle(
    model: SklearnModel,
    tau_by_size: dict[str, float],
    ranges: FeatureRanges,
    feature_names: tuple[str, ...],
    meta: dict,
    folder: str | Path,
) -> ModelBundle:
    """Write the model and its record into `folder` and return the bundle as loaded."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(model.estimator)
    (folder / MODEL_FILE).write_bytes(blob)
    meta = {**meta, "sklearn": sklearn.__version__}
    record = {
        "model_sha256": hashlib.sha256(blob).hexdigest(),
        "tau_by_size": tau_by_size,
        "ranges": dataclasses.asdict(ranges),
        "feature_names": list(feature_names),
        "meta": meta,
    }
    (folder / BUNDLE_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return load_bundle(folder, feature_names)


def load_bundle(folder: str | Path, feature_names: tuple[str, ...]) -> ModelBundle:
    """Read a bundle back; refuse a changed model, another library version or other features."""
    folder = Path(folder)
    record = json.loads((folder / BUNDLE_FILE).read_text())
    blob = (folder / MODEL_FILE).read_bytes()
    if hashlib.sha256(blob).hexdigest() != record["model_sha256"]:
        raise ValueError(f"{folder / MODEL_FILE} does not match the bundle's checksum")
    if record["meta"]["sklearn"] != sklearn.__version__:
        raise ValueError(
            f"the bundle was made with scikit-learn {record['meta']['sklearn']}, "
            f"this is {sklearn.__version__}"
        )
    if tuple(record["feature_names"]) != tuple(feature_names):
        raise ValueError("the bundle was made for other feature names or another order")
    ranges = record["ranges"]
    return ModelBundle(
        model=SklearnModel(pickle.loads(blob)),  # noqa: S301 - checked above; see the module note
        tau_by_size=dict(record["tau_by_size"]),
        ranges=FeatureRanges(
            tuple(ranges["lo"]), tuple(ranges["hi"]), tuple(ranges["may_be_missing"])
        ),
        feature_names=tuple(record["feature_names"]),
        model_sha256=record["model_sha256"],
        meta=record["meta"],
    )
