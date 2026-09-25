"""The SO-101 as the simulator loads it: the vendored model plus the points this project adds."""

import mujoco

# Gripper body frame, metres. Upstream's gripperframe sits on the fixed jaw's inner face, 7.9 mm off
# the roll axis; at gripper angle 0 the moving jaw's face mirrors it, so the pads' midpoint is on
# the roll axis at the same depth.
TCP_POS = (0.0, 0.0, -0.0981274)
# A half turn about x: z points out of the tool towards the fingertips, x towards the moving jaw.
TCP_QUAT = (0.0, 1.0, 0.0, 0.0)


def load_arm(mjcf):
    """Compile the vendored MJCF at `mjcf` with a TCP site added to the gripper body."""
    # MJCF cannot add a child to a body defined in an included file, so the site goes in through
    # MjSpec, MuJoCo's model-editing API, and the vendored file stays untouched.
    spec = mujoco.MjSpec.from_file(str(mjcf))
    spec.body("gripper").add_site(name="tcp", pos=TCP_POS, quat=TCP_QUAT)
    return spec.compile()
