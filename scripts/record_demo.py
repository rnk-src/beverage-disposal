"""Record a GIF of the sequencer carrying one container: to the pad, then to the bin or aside."""

import argparse
import sys
from pathlib import Path

import numpy as np

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.pad import PadSensor
from beverage_disposal.core.registry import load_registry
from beverage_disposal.core.safety_config import load_safety_limits
from beverage_disposal.core.sequencer import Item, Sequencer
from beverage_disposal.core.supervisor import RecordingIndicators, Supervisor
from beverage_disposal.sim.demo import FrameRecorder
from beverage_disposal.sim.skills import SimSkill

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
MODEL = ROOT / "assets" / "so101" / "so101_new_calib.xml"


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-ml", type=float, default=0.0)
    parser.add_argument("--container", default="can_12floz", help="registry type id")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "demo.gif")
    args = parser.parse_args(argv)

    scene = load_scene_config(CONFIGS / "scene.toml")
    fullness = load_fullness_config(CONFIGS / "fullness.toml")
    record = (registry := load_registry(CONFIGS / "containers.toml")).lookup(args.container)
    rng = np.random.default_rng(args.seed)
    frames = FrameRecorder()
    machine = Sequencer(
        scene=scene,
        registry=registry,
        fullness=fullness,
        supervisor=Supervisor(load_safety_limits(CONFIGS / "safety.toml"), RecordingIndicators()),
        skill=SimSkill(scene, MODEL, on_step=frames),
        pad=PadSensor(fullness.pad, rng),
    )
    item = Item(
        record.type_id,
        record.spec,
        args.volume_ml,
        1.0,
        record.spec.empty_mass,
        True if record.cap_check_required else None,
    )
    run = machine.process(item, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames.save(args.out)
    print(f"{run.destination}: wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
