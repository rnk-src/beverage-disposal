"""The sequencer with the simulated skill: two real episodes per container."""

import numpy as np
import pytest
from support.so101 import MJCF, SO101
from test_sequencer import BINNED, CAN, SCENE, SET_ASIDE_AREA, item, sequencer

from beverage_disposal.sim.skills import SimSkill


def simulated(seed=0):
    return sequencer(SimSkill(SCENE, SO101 / MJCF), seed=seed)


@pytest.mark.medium
@pytest.mark.parametrize(
    ("volume_ml", "destination"), [(0.0, BINNED), (355.0, SET_ASIDE_AREA)], ids=["empty", "full"]
)
def test_a_can_is_picked_to_the_pad_then_to_the_bin_or_the_set_aside_area(volume_ml, destination):
    machine, _, _ = simulated()
    run = machine.process(item(CAN, volume_ml), seed=1)
    assert run.destination == destination, run.events
    assert not run.misbinned


@pytest.mark.medium
def test_a_failing_second_leg_stops_the_run_and_never_bins(monkeypatch):
    machine, skill, _ = simulated()
    real = skill.skill.pick_and_place if hasattr(skill, "skill") else skill.pick_and_place
    calls = []

    def flaky(source, goal, **kwargs):
        calls.append(goal.name)
        result = real(source, goal, **kwargs)
        return result if len(calls) == 1 else type(result)(False, "forced")

    monkeypatch.setattr(skill, "pick_and_place", flaky)
    run = machine.process(item(CAN, 0.0), seed=1)
    assert run.destination == "stopped" and calls == ["pad", "bin"]
    assert np.isfinite(run.threshold_ml)
