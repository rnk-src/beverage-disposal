"""How accurate must the wall and liquid probes be to meet the 5 % target with zero misbins?

    uv run python scripts/sweep_probe_error.py
    uv run python scripts/sweep_probe_error.py --sizes pet_500ml pet_2000ml --trials 200

All runs are rule-only, noisy camera, acceptance stream, decision level, with assumed probe models
(no hardware): (a) no probes; (b) the wall probe at each wall error; (c) the liquid probe at each
liquid error; (d) both, over the whole grid of errors; (e) the liquid probe alone and both at the
errors of probes.toml, with the probe's reach set to each of REACHES_ML. (e) shows what a reach
costs in empties binned and saves in hidden contents (solid objects, under-reading probes) binned.
The results file is written to runs/ with its provenance.
"""

import argparse
import dataclasses
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from run_emptiness_grid import CONFIGS, PROMISE_ML, ROOT, run_grid

from beverage_disposal.core.bottle_family import load_family
from beverage_disposal.core.emptiness_eval import summarize_cells, target_met
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.perception import load_perception_noise
from beverage_disposal.core.policy import EmptinessPolicy, load_policy_config
from beverage_disposal.core.probes import ProbeModels, load_probe_models
from beverage_disposal.core.safety_config import load_safety_limits
from beverage_disposal.sim.harness import provenance

WALL_REL_SIGMAS = (0.02, 0.05, 0.10, 0.20)
LIQUID_SIGMAS_ML = (0.5, 1.0, 2.0, 4.0)
VOLUMES = (0.0, 10.0, 30.0, 31.0, 35.0, 50.0)
TARGET_VOLUMES = (0.0, 10.0)
MAX_SET_ASIDE = 0.05
REACHES_ML = (math.inf, 100.0, 60.0)


def plan(toml: ProbeModels) -> list[tuple[str, ProbeModels | None, float]]:
    """Every run as (label, probes, reach in mL)."""
    runs: list[tuple[str, ProbeModels | None, float]] = [("none", None, math.inf)]
    for sigma in WALL_REL_SIGMAS:
        wall = dataclasses.replace(toml.wall, rel_sigma=sigma)
        runs.append((f"wall rel_sigma={sigma:g}", ProbeModels(wall=wall), math.inf))
    for sigma in LIQUID_SIGMAS_ML:
        liquid = dataclasses.replace(toml.liquid, sigma_ml=sigma)
        runs.append((f"liquid sigma_ml={sigma:g}", ProbeModels(liquid=liquid), math.inf))
    for rel, sigma in itertools.product(WALL_REL_SIGMAS, LIQUID_SIGMAS_ML):
        wall = dataclasses.replace(toml.wall, rel_sigma=rel)
        liquid = dataclasses.replace(toml.liquid, sigma_ml=sigma)
        label = f"both rel_sigma={rel:g} sigma_ml={sigma:g}"
        runs.append((label, ProbeModels(wall, liquid), math.inf))
    for reach in REACHES_ML:
        runs.append(
            (f"liquid (probes.toml) reach={reach:g}", ProbeModels(liquid=toml.liquid), reach)
        )
        runs.append((f"both (probes.toml) reach={reach:g}", toml, reach))
    return runs


def total(counts: dict) -> int:
    """Binned bottles in a nested {..: [binned, trials]} summary."""
    if isinstance(counts, list):
        return counts[0]
    return sum(total(value) for value in counts.values())


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=["all"])
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "probe_sweep.json")
    args = parser.parse_args(argv)

    family = load_family(CONFIGS / "bottles.toml")
    sizes = list(family.classes) if args.sizes == ["all"] else args.sizes
    fullness = load_fullness_config(CONFIGS / "fullness.toml")
    pet = load_materials(CONFIGS / "materials.toml")["PET"]
    config = load_policy_config(CONFIGS / "policy.toml")
    payload_kg = load_safety_limits(CONFIGS / "safety.toml").max_container_mass
    noise = load_perception_noise(CONFIGS / "perception.toml")
    toml = load_probe_models(CONFIGS / "probes.toml")
    grid_args = SimpleNamespace(
        sizes=sizes, volumes=list(VOLUMES), trials=args.trials, seed=args.seed
    )

    results = []
    print(f"{'run':44s} met  empties@0  empties@10  misbin>=30  foreign  under-read")
    for label, models, reach in plan(toml):
        policy = EmptinessPolicy(
            fullness.rule,
            pet,
            family,
            dataclasses.replace(config, probe_reach_ml=reach),
            sizes=tuple(family.classes),  # neighbouring classes must be supported too
        )
        cells = run_grid(policy, family, pet, fullness.pad, noise, payload_kg, models, grid_args)
        summary = summarize_cells(cells, threshold_ml=PROMISE_ML)
        met = target_met(cells, volumes_ml=TARGET_VOLUMES, max_set_aside=MAX_SET_ASIDE)
        at = {
            volume: {
                c.size: c.binned
                for c in cells
                if c.volume_ml == volume
                and not c.deformed
                and not c.foreign_g
                and c.liquid_gain == 1.0
            }
            for volume in TARGET_VOLUMES
        }
        results.append(
            {
                "label": label,
                "probes": None if models is None else dataclasses.asdict(models),
                "probe_reach_ml": reach,
                "summary": summary,
                "target_met": met,
                "binned_at": {f"{v:g}": counts for v, counts in at.items()},
            }
        )
        print(
            f"{label:44s} {sum(met.values()):2d}/{len(met)}  "
            f"{sum(at[0.0].values()):6d}/{args.trials * len(sizes)}  "
            f"{sum(at[10.0].values()):6d}/{args.trials * len(sizes)}  "
            f"{summary['binned_at_or_over_threshold']:10d}  "
            f"{total(summary['foreign_binned']):7d}  {total(summary['under_reading_binned']):10d}",
            flush=True,
        )

    out = {
        "level": "decision only; rule-only; noisy camera; assumed probe models; no hardware",
        "seed": args.seed,
        "trials": args.trials,
        "sizes": sizes,
        "target": {"volumes_ml": TARGET_VOLUMES, "max_set_aside": MAX_SET_ASIDE},
        "runs": results,
        "provenance": provenance(fullness, family, pet, config, noise, toml),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
