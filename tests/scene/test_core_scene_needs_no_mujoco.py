"""The scene config and the container record are part of the core, so MuJoCo must not be needed."""

import subprocess
import sys

import pytest

WITHOUT_MUJOCO = """
import sys
sys.modules["mujoco"] = None
from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.containers import total_mass
config = load_scene_config(sys.argv[1])
print(total_mass(config.container, 355.0))
"""


def test_the_scene_config_loads_with_mujoco_unavailable():
    from support.paths import ROOT

    result = subprocess.run(
        [sys.executable, "-c", WITHOUT_MUJOCO, str(ROOT / "configs" / "scene.toml")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert float(result.stdout) == pytest.approx(0.369, abs=1e-12)
