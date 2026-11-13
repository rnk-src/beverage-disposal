"""Draws frames from running episodes and saves them as an animated GIF."""

from pathlib import Path

import mujoco
from PIL import Image

from beverage_disposal.sim.loop import ArmLoop

# A fixed view of the table from the front-left, aimed between the arm and the bin.
CAMERA = {"lookat": (0.25, 0.0, 0.05), "distance": 0.9, "azimuth": 140.0, "elevation": -30.0}
FRAME_MS = 50


class FrameRecorder:
    """Pass an instance as `on_step`: it keeps every `every`-th control step as a picture."""

    def __init__(self, every: int = 4, width: int = 480, height: int = 360):
        self._every = every
        self._size = (width, height)
        self._frames: list[Image.Image] = []
        self._count = 0
        self._renderer = None
        self._model = None

    def __call__(self, loop: ArmLoop) -> None:
        self._count += 1
        if (self._count - 1) % self._every:
            return
        if loop.model is not self._model:  # each episode builds a new model
            if self._renderer is not None:
                self._renderer.close()
            width, height = self._size
            self._renderer = mujoco.Renderer(loop.model, height, width)
            self._model = loop.model
        camera = mujoco.MjvCamera()
        camera.lookat[:] = CAMERA["lookat"]
        camera.distance = CAMERA["distance"]
        camera.azimuth = CAMERA["azimuth"]
        camera.elevation = CAMERA["elevation"]
        self._renderer.update_scene(loop.data, camera)
        self._frames.append(Image.fromarray(self._renderer.render()))

    def save(self, path: Path) -> None:
        if not self._frames:
            raise ValueError("no frames were recorded")
        first, *rest = self._frames
        first.save(path, save_all=True, append_images=rest, duration=FRAME_MS, loop=0)
