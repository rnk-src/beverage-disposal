"""The kinematics core runs without MuJoCo, as the ROS policy node will run it."""

import subprocess
import sys

# A module set to None in sys.modules fails every later import, as if it were not installed.
WITHOUT_MUJOCO = """
import sys
sys.modules["mujoco"] = None
from beverage_disposal.core.kinematics import HOME, fk
print(fk(HOME)[2, 3])
"""


def test_forward_kinematics_imports_and_runs_with_mujoco_unavailable():
    # Locked decision "ROS-free core": the ROS side gets NumPy and the chain file, not a simulator.
    result = subprocess.run([sys.executable, "-c", WITHOUT_MUJOCO], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert float(result.stdout) > 0


WITHOUT_MUJOCO_IK = """
import sys
sys.modules["mujoco"] = None
import numpy as np
from beverage_disposal.core.ik import solve_ik
from beverage_disposal.core.kinematics import HOME, fk
print(solve_ik(fk(HOME)[:3, 3] + [0.01, 0, 0]).converged)
"""


def test_inverse_kinematics_imports_and_runs_with_mujoco_unavailable():
    # The ROS policy node runs the IK too (locked decision "ROS-free core").
    result = subprocess.run(
        [sys.executable, "-c", WITHOUT_MUJOCO_IK], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


WITHOUT_MUJOCO_CONTROLLER = """
import sys
sys.modules["mujoco"] = None
import numpy as np
from beverage_disposal.core.controller import EEDeltaController
from beverage_disposal.core.kinematics import HOME
controller = EEDeltaController()
controller.reset(HOME, 0.5)
print(controller.step(np.array([1.0, 0, 0, 0]), HOME, 0.5).ik_ok)
"""


def test_the_controller_imports_and_runs_with_mujoco_unavailable():
    result = subprocess.run(
        [sys.executable, "-c", WITHOUT_MUJOCO_CONTROLLER], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"
