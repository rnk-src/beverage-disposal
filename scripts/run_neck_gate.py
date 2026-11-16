"""Run the grasp gate for the bottle sizes and write the results file.

    uv run python scripts/run_neck_gate.py                # every size, neck grasp and body grasp
    uv run python scripts/run_neck_gate.py --sizes pet_500ml --modes neck

Each size is carried from the pad to the set-aside area 12 times empty and 12 times at the
payload-limit fill, with the scripted controller. A size out of reach is listed, not run. The
pitch is the steepest one that reaches every pose of the size's tallest and shortest bottle.
"""

import argparse
import dataclasses
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from beverage_disposal.core.bottle_family import GRASP_MODES, load_family
from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.gate import load_gate_bar, passes
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.safety_config import load_safety_limits
from beverage_disposal.sim.harness import provenance
from beverage_disposal.sim.neck_gate import FILLS, class_pitch_deg, run_gate

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
MODEL = ROOT / "assets" / "so101" / "so101_new_calib.xml"
MAX_PITCH_DEG = 60  # the scene's grasp pitch; a bottle is never taken steeper than the can


def _configs():
    return (
        load_scene_config(CONFIGS / "scene.toml"),
        load_family(CONFIGS / "bottles.toml"),
        load_materials(CONFIGS / "materials.toml")["PET"],
        load_gate_bar(CONFIGS / "gate.toml"),
        load_safety_limits(CONFIGS / "safety.toml"),
    )


def _one(job):
    name, mode, fill = job
    scene, family, pet, bar, limits = _configs()
    size = dataclasses.replace(family.classes[name], grasp_mode=mode)
    pitch = class_pitch_deg(scene, family, pet, size)
    if pitch is None:
        return {"size": name, "mode": mode, "fill": fill, "pitch_deg": None, "passed": False}
    size = dataclasses.replace(size, grasp_pitch_deg=float(min(pitch, MAX_PITCH_DEG)))
    row = run_gate(scene, MODEL, family, pet, size, fill, bar, payload_kg=limits.max_container_mass)
    return {
        "size": name,
        "mode": mode,
        "fill": fill,
        "pitch_deg": size.grasp_pitch_deg,
        "slips_m": list(row.slips),
        "settle_times_s": list(row.settle_times),
        "places_upright": list(row.places_upright),
        "passed": passes(row, bar),
    }


def main(argv):
    scene, family, pet, bar, limits = _configs()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=list(family.classes))
    parser.add_argument("--modes", nargs="+", default=list(GRASP_MODES), choices=GRASP_MODES)
    parser.add_argument("--jobs", type=int, default=6, help="rows run at the same time")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "neck_gate.json")
    args = parser.parse_args(argv)

    jobs = [(size, mode, fill) for size in args.sizes for mode in args.modes for fill in FILLS]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(_one, jobs))
    results = {
        "bar": dataclasses.asdict(bar),
        "rows": rows,
        "provenance": provenance(scene, family, pet, bar, limits),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")

    print("size         mode  fill     pitch  places  worst slip  worst settle  pass")
    for row in rows:
        if row["pitch_deg"] is None:
            print(f"{row['size']:12s} {row['mode']:5s} {row['fill']:8s} out of reach")
            continue
        places = sum(row["places_upright"])
        print(
            f"{row['size']:12s} {row['mode']:5s} {row['fill']:8s} {row['pitch_deg']:5.0f}  "
            f"{places:2d}/{len(row['places_upright'])}   {max(row['slips_m']) * 1000:7.1f} mm  "
            f"{max(row['settle_times_s']):9.2f} s   {'yes' if row['passed'] else 'no'}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
