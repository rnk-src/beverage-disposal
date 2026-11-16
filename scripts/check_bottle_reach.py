"""Print, per bottle size, whether the arm reaches every pose of a neck grasp from the pad.

    uv run python scripts/check_bottle_reach.py

Pure kinematics, no physics: it takes the tallest and the shortest of 200 drawn bottles per size
and reports the steepest pitch that reaches each pose, and the steepest one that reaches them all.
"""

from pathlib import Path

import numpy as np

from beverage_disposal.core.bottle_family import container_spec_for, load_family, sample_bottle
from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.reach import container_poses, reach_for_sizes, steepest_common_pitch

ROOT = Path(__file__).resolve().parents[1]
DRAWS = 200


def main() -> None:
    scene = load_scene_config(ROOT / "configs" / "scene.toml")
    family = load_family(ROOT / "configs" / "bottles.toml")
    pet = load_materials(ROOT / "configs" / "materials.toml")["PET"]
    can = scene.container
    print("size         bottle    height  grasp z  common  steepest pitch per pose (deg)")
    for name, size in family.classes.items():
        specs = [
            container_spec_for(
                sample_bottle(family, name, pet, np.random.default_rng(seed)),
                size,
                friction=can.friction,
                torsional_friction=can.torsional_friction,
            )
            for seed in range(DRAWS)
        ]
        ends = {
            "tallest": max(specs, key=lambda spec: spec.height),
            "shortest": min(specs, key=lambda spec: spec.height),
        }
        both = [pose for spec in ends.values() for pose in container_poses(scene, spec)]
        common = steepest_common_pitch(both)
        for which, rows in reach_for_sizes(scene, ends).items():
            spec = ends[which]
            pitches = "  ".join(
                f"{row.label}={'none' if row.best_pitch_deg is None else row.best_pitch_deg}"
                for row in rows
            )
            print(
                f"{name:12s} {which:9s} {spec.height:.3f}   {spec.grasp_height:.3f}    "
                f"{'none' if common is None else common:>4}    {pitches}"
            )


if __name__ == "__main__":
    main()
