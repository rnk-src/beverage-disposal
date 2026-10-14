"""Measure the arm's servo step responses in MuJoCo and write the reference file for Gazebo."""

import argparse
import json
import sys
from pathlib import Path

from beverage_disposal.sim.actuator_response import measure_step_responses
from beverage_disposal.sim.arm import load_arm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "assets" / "so101" / "so101_new_calib.xml"
DEFAULT_OUT = ROOT / "configs" / "actuator_step_response.json"


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MJCF file to read")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON file to write")
    args = parser.parse_args(argv)
    args.out.write_text(json.dumps(measure_step_responses(load_arm(args.model)), indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
