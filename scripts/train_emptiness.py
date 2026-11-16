"""Fit and calibrate the emptiness classifier on simulated bottles and save the bundle.

    uv run python scripts/train_emptiness.py

Writes runs/emptiness_model/ (model.pkl and bundle.json) and prints the threshold per size and the
calibration rows per cell. Decision level only: no arm moves. The sizes default to every size in
bottles.toml, whether or not its grasp gate has passed.
"""

import argparse
import sys
from pathlib import Path

from beverage_disposal.core.bottle_family import load_family
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.np_calibration import min_samples
from beverage_disposal.core.perception import load_perception_noise
from beverage_disposal.core.policy import EmptinessPolicy, load_policy_config
from beverage_disposal.core.safety_config import load_safety_limits
from beverage_disposal.train.emptiness.dataset import DatasetConfig
from beverage_disposal.train.emptiness.fit import FitConfig
from beverage_disposal.train.emptiness.pipeline import train_bundle

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
# With the least sample size (598) a threshold is its cell's largest score, far stricter than
# alpha asks. With 5000 rows it is about the 17th largest, close to the 99.5th percentile.
CALIBRATION_ROWS = 5000
DRAWS_PER_ROW = 400  # ambiguous bottles are rare for small sizes; give up on a cell after this


def main(argv):
    family = load_family(CONFIGS / "bottles.toml")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=list(family.classes))
    parser.add_argument("--rows-per-size", type=int, default=40_000, help="training bottles")
    parser.add_argument(
        "--rows-per-cell", type=int, default=CALIBRATION_ROWS, help="calibration bottles per cell"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "emptiness_model")
    args = parser.parse_args(argv)

    fullness = load_fullness_config(CONFIGS / "fullness.toml")
    config = load_policy_config(CONFIGS / "policy.toml")
    policy = EmptinessPolicy(
        fullness.rule,
        load_materials(CONFIGS / "materials.toml")["PET"],
        family,
        config,
        sizes=tuple(args.sizes),
    )
    needed = min_samples(config.alpha, config.delta)
    data = DatasetConfig(
        sizes=tuple(args.sizes),
        rows_per_size=args.rows_per_size,
        volume_bins=config.volume_bins,
        rows_per_cell=max(needed, args.rows_per_cell),
        max_draws_per_cell=DRAWS_PER_ROW * max(needed, args.rows_per_cell),
        payload_kg=load_safety_limits(CONFIGS / "safety.toml").max_container_mass,
    )
    noise = load_perception_noise(CONFIGS / "perception.toml")
    bundle, report = train_bundle(
        policy, fullness.pad, noise, data, FitConfig(), args.out, seed=args.seed
    )

    print(f"trained on {report.train_rows} bottles; alpha {config.alpha}, delta {config.delta}")
    print(f"calibration rows per cell (at least {needed} needed):")
    for (size, low), count in report.cell_rows.items():
        print(f"  {size:12s} from {low:g} mL: {count}")
    print("threshold per size (a score must be above it to bin):")
    for size, tau in bundle.tau_by_size.items():
        print(f"  {size:12s} {tau:.6f}")
    if report.empty_cells:
        print(f"cells with no ambiguous bottle: {report.empty_cells}")
    if report.uncalibrated_sizes:
        print(f"sizes that can never be overturned: {report.uncalibrated_sizes}")
    print(f"model sha256 {bundle.model_sha256}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
