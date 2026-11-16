"""Run the emptiness policy over a grid of bottle sizes and volumes and write the results file.

    uv run python scripts/run_emptiness_grid.py --policy rule-only --sizes all --trials 1000
    uv run python scripts/run_emptiness_grid.py --policy full --sizes all --trials 1000

`--probes wall|liquid|both` fits the simulated wall and liquid probes of configs/probes.toml (an
experiment: no hardware, every figure assumed). With the liquid probe the grid also runs bottles
it reads low or cannot see, which are expected to be binned: they show the failure to expect.

Decision level only: bottles are drawn, weighed on the pad model, seen through the camera noise
model and judged; no arm moves. Each size is also tested at its payload fill, with crushed bottles
and with a solid object in an empty bottle. The bottles come from the acceptance streams, which
share nothing with the classifier's training and calibration bottles. `--policy full` loads the
classifier's bundle and also runs the rule-only policy on the same bottles, to compare. The sizes
default to every size the arm can reach, whether or not its grasp gate has passed, so the file
says which of them are supported.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from beverage_disposal.core.bottle_family import load_family
from beverage_disposal.core.emptiness_eval import misbin_grid, payload_volume_ml, summarize_cells
from beverage_disposal.core.features import feature_names
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.materials import load_materials
from beverage_disposal.core.perception import load_perception_noise
from beverage_disposal.core.policy import EmptinessPolicy, load_policy_config
from beverage_disposal.core.probes import ProbeModels, load_probe_models
from beverage_disposal.core.safety_config import load_safety_limits
from beverage_disposal.sim.harness import provenance
from beverage_disposal.train.emptiness.bundle import load_bundle

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DEFAULT_VOLUMES = (0.0, 10.0, 20.0, 29.0, 30.0, 31.0, 35.0, 50.0)
# The sizes the arm reached in 2B.3; the larger ones are out of its reach.
REACHABLE = ("pet_250ml", "pet_330ml", "pet_500ml")
PROMISE_ML = 30.0
# The acceptance split of the dataset streams, then a stream per kind of cell.
ACCEPTANCE = (3, 2)
SOUND, CRUSHED, FOREIGN = 0, 1, 2
CRUSHED_VOLUMES = (0.0, 35.0)
FOREIGN_G = (50.0, 100.0)  # a solid object in an empty bottle
UNDER_READ = 4  # a stream index after FOREIGN + 1
UNDER_READ_GAINS = (0.5, 0.0)  # 0.5: a probe that reads low; 0.0: frozen or slushed contents
UNDER_READ_VOLUMES = (30.0, 50.0, 100.0, 200.0)


def run_grid(policy, family, pet, pad, noise, payload_kg, models, args):
    """The sound cells per size, payload fill included, then the crushed and foreign cells."""
    common = dict(trials=args.trials, seed=args.seed, noise=noise, probes=models)
    cells = []
    for size in args.sizes:
        one = dict(common, sizes=(size,))
        top = payload_volume_ml(family, size, payload_kg=payload_kg)
        volumes = tuple(sorted({*args.volumes, top}))
        cells += misbin_grid(
            policy, family, pet, pad, volumes_ml=volumes, stream=(*ACCEPTANCE, SOUND), **one
        )
        if noise is None:
            continue  # a crushed bottle needs the camera model, and the level check a seen level
        cells += misbin_grid(
            policy, family, pet, pad, volumes_ml=CRUSHED_VOLUMES, stream=(*ACCEPTANCE, CRUSHED),
            deformed=True, **one,
        )  # fmt: skip
        for index, grams in enumerate(FOREIGN_G):
            cells += misbin_grid(
                policy, family, pet, pad, volumes_ml=(0.0,), foreign_g=grams,
                stream=(*ACCEPTANCE, FOREIGN + index), **one,
            )  # fmt: skip
        if models is None or models.liquid is None:
            continue
        for index, gain in enumerate(UNDER_READ_GAINS):
            cells += misbin_grid(
                policy, family, pet, pad, volumes_ml=UNDER_READ_VOLUMES, liquid_gain=gain,
                stream=(*ACCEPTANCE, UNDER_READ + index), **one,
            )  # fmt: skip
    return cells


def probe_models(args) -> ProbeModels | None:
    """The probes the flags ask for, with any error override, or None for no probes."""
    if args.probes == "none":
        return None
    models = load_probe_models(CONFIGS / "probes.toml")
    wall, liquid = models.wall, models.liquid
    if args.wall_rel_sigma is not None:
        wall = dataclasses.replace(wall, rel_sigma=args.wall_rel_sigma)
    if args.liquid_sigma_ml is not None:
        liquid = dataclasses.replace(liquid, sigma_ml=args.liquid_sigma_ml)
    return ProbeModels(
        wall=wall if args.probes in ("wall", "both") else None,
        liquid=liquid if args.probes in ("liquid", "both") else None,
    )


def main(argv):
    family = load_family(CONFIGS / "bottles.toml")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", choices=["rule-only", "full"], default="rule-only")
    parser.add_argument("--bundle", type=Path, default=ROOT / "runs" / "emptiness_model")
    parser.add_argument("--sizes", nargs="+", default=list(REACHABLE), help="size ids, or all")
    parser.add_argument("--volumes", type=float, nargs="+", default=list(DEFAULT_VOLUMES))
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exact-camera", action="store_true", help="skip the camera noise of perception.toml"
    )
    parser.add_argument(
        "--probes", choices=["none", "wall", "liquid", "both"], default="none",
        help="fit the simulated probes of probes.toml",
    )  # fmt: skip
    parser.add_argument("--wall-rel-sigma", type=float, default=None)
    parser.add_argument("--liquid-sigma-ml", type=float, default=None)
    parser.add_argument("--probe-reach-ml", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.policy == "full" and args.probes != "none":
        parser.error(
            "--policy full cannot be combined with --probes: the classifier never saw them"
        )
    if args.sizes == ["all"]:
        args.sizes = list(family.classes)

    fullness = load_fullness_config(CONFIGS / "fullness.toml")
    pet = load_materials(CONFIGS / "materials.toml")["PET"]
    config = load_policy_config(CONFIGS / "policy.toml")
    if args.probe_reach_ml is not None:
        config = dataclasses.replace(config, probe_reach_ml=args.probe_reach_ml)
    models = probe_models(args)
    payload_kg = load_safety_limits(CONFIGS / "safety.toml").max_container_mass
    noise = None if args.exact_camera else load_perception_noise(CONFIGS / "perception.toml")
    rule_only = EmptinessPolicy(fullness.rule, pet, family, config, sizes=tuple(args.sizes))
    grid = (family, pet, fullness.pad, noise, payload_kg, models, args)

    results = {
        "policy": args.policy,
        "level": "decision only: no arm, no physics",
        "grasp_gate_passed": {size: size in family.supported() for size in args.sizes},
        "camera": "exact" if noise is None else "noisy (perception.toml)",
        "seed": args.seed,
        "probes": None if models is None else dataclasses.asdict(models),
        "probe_reach_ml": config.probe_reach_ml,
        "streams": "acceptance: disjoint from the training and calibration bottles",
    }
    if args.policy == "full":
        bundle = load_bundle(args.bundle, feature_names(family))
        policy = dataclasses.replace(rule_only, bundle=bundle)
        results["bundle"] = {
            "model_sha256": bundle.model_sha256,
            "tau_by_size": bundle.tau_by_size,
            "meta": bundle.meta,
        }
        results["allow_overturn"] = config.allow_overturn
        results["rule_only_summary"] = summarize_cells(
            run_grid(rule_only, *grid), threshold_ml=PROMISE_ML
        )
    else:
        policy = rule_only
    cells = run_grid(policy, *grid)
    summary = summarize_cells(cells, threshold_ml=PROMISE_ML)
    results["summary"] = summary
    results["cells"] = [dataclasses.asdict(cell) for cell in cells]
    results["provenance"] = provenance(fullness, family, pet, config, noise, models)
    name = f"emptiness_{args.policy.replace('-', '_')}"
    if models is not None:
        name += f"_probes_{args.probes}"
    out = args.out or ROOT / "runs" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")

    print("size          volume  kind         binned      upper 95 %  overturned  zones")
    for cell in cells:
        kind = (
            "crushed" if cell.deformed else f"object {cell.foreign_g:g} g" if cell.foreign_g else ""
        )
        print(
            f"{cell.size:12s} {cell.volume_ml:6g}  {kind:12s} {cell.binned:5d}/{cell.trials:<5d} "
            f"{cell.upper_95 * 100:8.3f} %  {cell.overturned:10d}  {cell.zones}"
        )
    misbinned = summary["binned_at_or_over_threshold"]
    worst = summary["worst_upper_95_at_or_over_threshold"]
    print(
        f"in simulation, binned at {PROMISE_ML:g} mL or more: {misbinned} of "
        f"{summary['trials_at_or_over_threshold']}; worst cell's Wilson 95 % upper bound "
        f"{worst * 100:.3f} %"
    )
    print(f"empties binned: {summary['empties_binned']}")
    if args.policy == "full":
        print(f"rule-only, same bottles: {results['rule_only_summary']['empties_binned']}")
    print(f"overturned: {summary['overturned']}; vetoes: {summary['vetoes']}")
    print(f"crushed binned: {summary['crushed_binned']}")
    print(f"foreign objects binned: {summary['foreign_binned']}")
    print(f"probe under-reading binned (not counted above): {summary['under_reading_binned']}")
    print(f"wrote {out}")
    return 1 if misbinned else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
