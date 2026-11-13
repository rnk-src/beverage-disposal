"""The demo recorder: frames drawn from a running episode, saved as an animated GIF."""

import pytest
from PIL import Image
from support.so101 import MJCF, SO101
from test_sequencer import CAN, SCENE, item, sequencer

from beverage_disposal.sim.demo import FrameRecorder
from beverage_disposal.sim.skills import SimSkill


@pytest.mark.medium
def test_a_demo_of_a_whole_run_is_a_gif_with_many_distinct_frames(tmp_path):
    frames = FrameRecorder(every=20, width=160, height=120)
    machine, _, _ = sequencer(SimSkill(SCENE, SO101 / MJCF, on_step=frames))
    machine.process(item(CAN, 0.0), seed=1)
    path = tmp_path / "demo.gif"
    frames.save(path)
    with Image.open(path) as gif:
        assert gif.n_frames >= 10
        first = gif.convert("RGB").tobytes()
        gif.seek(gif.n_frames - 1)
        assert gif.convert("RGB").tobytes() != first  # the scene moved


def test_saving_with_no_frames_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        FrameRecorder().save(tmp_path / "x.gif")
