"""Look at the vendored SO-101 arm: an interactive viewer, or a PNG snapshot with no window."""

import argparse
import sys
from pathlib import Path

import mujoco
from PIL import Image

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "assets" / "so101" / "so101_new_calib.xml"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MJCF file to load")
    parser.add_argument(
        "--snapshot",
        type=Path,
        metavar="PNG",
        help="write a picture of the arm at rest to this file and exit, without opening a window",
    )
    parser.add_argument("--width", type=int, default=640, help="snapshot width in pixels")
    parser.add_argument("--height", type=int, default=480, help="snapshot height in pixels")
    return parser.parse_args(argv)


def load(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    # A fresh MjData has no computed body positions, so it would render as an empty frame.
    mujoco.mj_forward(model, data)
    return model, data


def snapshot(model, data, path, width, height):
    with mujoco.Renderer(model, height, width) as renderer:
        renderer.update_scene(data)
        pixels = renderer.render()
    Image.fromarray(pixels).save(path)


def main(argv=None, launch=None):
    args = parse_args(argv)
    try:
        model, data = load(args.model)
    except ValueError as error:
        print(f"error: cannot load {args.model}: {error}", file=sys.stderr)
        return 1
    if args.snapshot:
        snapshot(model, data, args.snapshot, args.width, args.height)
        print(f"wrote {args.snapshot}")
        return 0
    if launch is None:
        import mujoco.viewer

        launch = mujoco.viewer.launch
    launch(model, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
