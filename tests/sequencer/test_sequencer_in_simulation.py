"""The sequencer with the simulated skill: one real episode per container, from the pad."""

import numpy as np
import pytest
from support.so101 import MJCF, SO101
from test_sequencer import BINNED, CAN, FAMILY, SCENE, SET_ASIDE_AREA, item, pet_item, sequencer

from beverage_disposal.sim.skills import SimSkill


def simulated(seed=0):
    return sequencer(SimSkill(SCENE, SO101 / MJCF), seed=seed)


@pytest.mark.medium
@pytest.mark.parametrize(
    ("volume_ml", "destination"), [(0.0, BINNED), (355.0, SET_ASIDE_AREA)], ids=["empty", "full"]
)
def test_a_can_on_the_pad_is_carried_to_the_bin_or_the_set_aside_area(volume_ml, destination):
    machine, _, _ = simulated()
    run = machine.process(item(CAN, volume_ml), seed=1)
    assert run.destination == destination, run.events
    assert not run.misbinned


@pytest.mark.medium
def test_a_failing_carry_stops_the_run_and_never_bins(monkeypatch):
    machine, skill, _ = simulated()
    real = skill.pick_and_place
    calls = []

    def failing(source, goal, **kwargs):
        calls.append(goal.name)
        return type(real(source, goal, **kwargs))(False, "forced")

    monkeypatch.setattr(skill, "pick_and_place", failing)
    run = machine.process(item(CAN, 0.0), seed=1)
    assert run.destination == "stopped" and calls == ["bin"]
    assert np.isfinite(run.threshold_ml)


@pytest.mark.medium
def test_a_bottle_item_is_simulated_as_the_bottle_and_not_as_a_cylinder():
    skill = SimSkill(SCENE, SO101 / MJCF, family=FAMILY)
    bottle = pet_item(volume_ml=0.0, size="pet_250ml")
    assert skill.scene_for(bottle).bottle is bottle.bottle
    assert skill.scene_for(item(CAN, 0.0)).bottle is None
    with pytest.raises(ValueError, match="family"):
        SimSkill(SCENE, SO101 / MJCF).scene_for(bottle)
