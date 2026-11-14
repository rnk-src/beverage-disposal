"""The evaluation harness: seeded runs of the sequencer over a grid of volumes, and the record."""

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

from beverage_disposal.core.acceptance import stop_of, summarize
from beverage_disposal.core.coinft import CoinFT
from beverage_disposal.core.pad import PadSensor
from beverage_disposal.core.sequencer import Item, Sequencer, Skill
from beverage_disposal.core.supervisor import RecordingIndicators, Supervisor

TRACKED_PACKAGES = ("mujoco", "numpy", "gymnasium")


def evaluate(
    skill: Skill,
    *,
    volumes_ml,
    trials: int,
    container,
    scene,
    registry,
    fullness,
    tilt,
    limits,
    density: float = 1.0,
    known: bool = True,
    use_tilt: bool = True,
) -> dict:
    """Run `trials` seeded episodes per volume (seeds 0..trials-1) and return the results record.

    Each run gets a fresh supervisor and sensors seeded from the run's seed, so a run does not
    depend on the ones before it.
    """
    runs = []
    rows = []
    seeds = {}
    for volume in volumes_ml:
        seeds[f"{volume:g}"] = list(range(trials))
        for seed in range(trials):
            rng = np.random.default_rng([seed, int(volume * 10)])
            machine = Sequencer(
                scene=scene,
                registry=registry,
                fullness=fullness,
                tilt=tilt,
                supervisor=Supervisor(limits, RecordingIndicators()),
                skill=skill,
                pad=PadSensor(fullness.pad, rng),
                coinft=CoinFT(tilt.sensor, rng),
                use_tilt=use_tilt,
            )
            item = Item(
                type_id=container.type_id if known else None,
                spec=container.spec,
                volume_ml=volume,
                density=density,
                empty_mass_kg=container.spec.empty_mass,
                cap_confirmed=True if container.cap_check_required else None,
            )
            run = machine.process(item, seed=seed)
            runs.append(run)
            rows.append(
                {
                    "volume_ml": volume,
                    "seed": seed,
                    "destination": run.destination,
                    "stop": stop_of(run),
                }
            )
    return {
        "summary": summarize(runs),
        "seeds": seeds,
        "runs": rows,
        "provenance": provenance(scene, registry, fullness, tilt, limits),
    }


def provenance(*configs) -> dict:
    """What is needed to reproduce a result: the commit, a hash of the configs and the versions."""
    root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = "unknown", False
    return {
        "git_sha": sha,
        "tracked_files_modified": dirty,
        "config_hash": hashlib.sha256(repr(configs).encode()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            **{name: metadata.version(name) for name in TRACKED_PACKAGES},
        },
        "platform": sys.platform,
    }
