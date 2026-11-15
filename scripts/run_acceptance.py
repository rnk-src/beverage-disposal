"""Run the sequencer in simulation over a grid of volumes and write the results file."""

import argparse
import json
import sys
from pathlib import Path

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.registry import load_registry
from beverage_disposal.core.safety_config import load_safety_limits
from beverage_disposal.sim.harness import evaluate
from beverage_disposal.sim.skills import SimSkill

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
MODEL = ROOT / "assets" / "so101" / "so101_new_calib.xml"
# 381 trials with no miss bound a rate under 1 % at 95 % confidence (Wilson).
DEFAULT_TRIALS = 381
DEFAULT_VOLUMES = (0.0, 30.0, 31.0, 35.0)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="runs per volume")
    parser.add_argument("--volumes", type=float, nargs="+", default=DEFAULT_VOLUMES, help="mL")
    parser.add_argument("--container", default="can_12floz", help="registry type id")
    parser.add_argument("--unknown", action="store_true", help="present it as an unknown type")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "acceptance.json")
    args = parser.parse_args(argv)

    scene = load_scene_config(CONFIGS / "scene.toml")
    registry = load_registry(CONFIGS / "containers.toml")
    results = evaluate(
        SimSkill(scene, MODEL),
        volumes_ml=args.volumes,
        trials=args.trials,
        container=registry.lookup(args.container),
        scene=scene,
        registry=registry,
        fullness=load_fullness_config(CONFIGS / "fullness.toml"),
        limits=load_safety_limits(CONFIGS / "safety.toml"),
        known=not args.unknown,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
