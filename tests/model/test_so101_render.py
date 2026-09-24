"""The vendored SO-101 draws offscreen: a visible arm, in the requested size, following its pose."""

import mujoco
import numpy as np
from so101 import MJCF, SO101
from support.images import foreground_fraction

WIDTH, HEIGHT = 320, 240


def load():
    model = mujoco.MjModel.from_xml_path(str(SO101 / MJCF))
    return model, mujoco.MjData(model)


def load_at_rest():
    """A state whose body positions are computed: a fresh MjData draws nothing (tested below)."""
    model, data = load()
    mujoco.mj_forward(model, data)
    return model, data


def render(model, data):
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        renderer.update_scene(data)
        return renderer.render()


def test_a_render_has_the_requested_size_and_is_eight_bit_color():
    picture = render(*load_at_rest())
    assert picture.shape == (HEIGHT, WIDTH, 3)
    assert picture.dtype == np.uint8


def test_the_arm_is_visible_and_does_not_fill_the_frame():
    # A blank frame is 0%; a 0.3 m arm in a default view is a few percent (measured 6.4%).
    assert 0.01 < foreground_fraction(render(*load_at_rest())) < 0.5


def test_a_fresh_state_renders_blank_until_the_kinematics_are_computed():
    # MuJoCo draws from the computed body positions, which are all zero in a fresh MjData.
    assert foreground_fraction(render(*load())) == 0.0


def test_the_render_follows_the_arms_pose():
    model, data = load_at_rest()
    before = render(model, data)
    data.joint("shoulder_pan").qpos[0] = 1.0
    mujoco.mj_forward(model, data)
    after = render(model, data)
    changed = (np.abs(before.astype(int) - after.astype(int)).sum(axis=2) > 30).mean()
    assert changed > 0.01
