"""Run the emptiness policy over a grid of bottle sizes and volumes and write the results file.

    uv run python scripts/run_emptiness_grid.py --policy rule-only --trials 1000

Decision level only: bottles are drawn, weighed on the pad model, seen through the camera noise
model and judged; no arm moves. The
sizes default to every size the arm can reach, whether or not its grasp gate has passed, so the
file says which of them are supported.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from beverage_disposal.core.bottle_family import load_family
from beverage_disposal.core.emptiness_eval import misbin_grid
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.perception import load_perception_noise
from beverage_disposal.core.policy import EmptinessPolicy, load_policy_config
from beverage_disposal.sim.harness import provenance

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DEFAULT_VOLUMES = (0.0, 10.0, 20.0, 29.0, 30.0, 31.0, 35.0, 50.0)
# The sizes the arm reached in 2B.3; the larger ones are out of its reach.
REACHABLE = ("pet_250ml", "pet_330ml", "pet_500ml")
PROMISE_ML = 30.0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", choices=["rule-only"], default="rule-only")
    parser.add_argument("--sizes", nargs="+", default=list(REACHABLE))
    parser.add_argument("--volumes", type=float, nargs="+", default=list(DEFAULT_VOLUMES))
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exact-camera", action="store_true", help="skip the camera noise of perception.toml"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    fullness = load_fullness_config(CONFIGS / "fullness.toml")
    family = load_family(CONFIGS / "bottles.toml")
    pet = load_materials(CONFIGS / "materials.toml")["PET"]
    config = load_policy_config(CONFIGS / "policy.toml")
    noise = None if args.exact_camera else load_perception_noise(CONFIGS / "perception.toml")
    policy = EmptinessPolicy(fullness.rule, pet, family, config, sizes=tuple(args.sizes))
    cells = misbin_grid(
        policy,
        family,
        pet,
        fullness.pad,
        sizes=tuple(args.sizes),
        volumes_ml=tuple(args.volumes),
        trials=args.trials,
        seed=args.seed,
        noise=noise,
    )
    results = {
        "policy": args.policy,
        "level": "decision only: no arm, no physics",
        "grasp_gate_passed": {size: size in family.supported() for size in args.sizes},
        "camera": "exact" if noise is None else "noisy (perception.toml)",
        "seed": args.seed,
        "cells": [dataclasses.asdict(cell) for cell in cells],
        "provenance": provenance(fullness, family, pet, config, noise),
    }
    out = args.out or ROOT / "runs" / f"emptiness_{args.policy.replace('-', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")

    print("size          volume  binned      upper 95 %  zones")
    for cell in cells:
        print(
            f"{cell.size:12s} {cell.volume_ml:6g}  {cell.binned:5d}/{cell.trials:<5d} "
            f"{cell.upper_95 * 100:8.3f} %  {cell.zones}"
        )
    misbinned = [cell for cell in cells if cell.volume_ml >= PROMISE_ML and cell.binned]
    print(f"binned at {PROMISE_ML:g} mL or more: {sum(cell.binned for cell in misbinned)}")
    print(f"wrote {out}")
    return 1 if misbinned else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
