"""The MuJoCo closed loop: joint targets in, one 20 Hz control step of physics, measurements out."""

import mujoco

# Locked decision: 20 Hz policy rate, with frame skip over the physics steps.
CONTROL_PERIOD = 0.05
ARM_JOINTS = 5


class ArmLoop:
    def __init__(self, model):
        self._model = model
        self._data = mujoco.MjData(model)
        # Derived from the timestep, so halving the physics step keeps the control period the same.
        self._frame_skip = round(CONTROL_PERIOD / model.opt.timestep)
        self._base = model.body("base").id
        self._tcp = model.site("tcp").id

    def reset(self, joints, gripper):
        """Rest the arm in the given pose, holding it there (the targets equal the pose)."""
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[:] = [*joints, gripper]
        self._data.ctrl[:] = [*joints, gripper]
        mujoco.mj_forward(self._model, self._data)

    def step(self, joint_targets, gripper_target, on_substep=None):
        """Hold these targets for one control period; `on_substep` runs after every physics step."""
        self._data.ctrl[:] = [*joint_targets, gripper_target]
        for _ in range(self._frame_skip):
            mujoco.mj_step(self._model, self._data)
            if on_substep is not None:
                on_substep()

    @property
    def time(self):
        return self._data.time

    @property
    def joints(self):
        return self._data.qpos[:ARM_JOINTS].copy()

    @property
    def gripper(self):
        return float(self._data.qpos[ARM_JOINTS])

    @property
    def joint_velocities(self):
        return self._data.qvel[:ARM_JOINTS].copy()

    @property
    def tip(self):
        """The TCP in the arm's base frame, as fk reports it."""
        base_pos = self._data.xpos[self._base]
        base_rot = self._data.xmat[self._base].reshape(3, 3)
        return base_rot.T @ (self._data.site_xpos[self._tcp] - base_pos)
