"""The pick-and-place skill as the simulation runs it: the scripted controller in MuJoCo."""

import dataclasses

from beverage_disposal.core.config import SceneConfig
from beverage_disposal.core.layout import Goal
from beverage_disposal.core.scripted import ScriptedController
from beverage_disposal.core.sequencer import Item, Placement, SkillResult
from beverage_disposal.sim.rollout import run_episode


class SimSkill:
    """Runs one leg as a full simulated episode. Each leg starts with the arm at its ready pose."""

    def __init__(self, scene: SceneConfig, mjcf):
        self._scene = scene
        self._mjcf = mjcf

    def pick_and_place(
        self, source: Placement | None, goal: Goal, *, item: Item, seed: int
    ) -> SkillResult:
        # The simulation's liquid weighs a gram per millilitre: a denser drink counts as more of it.
        config = dataclasses.replace(
            self._scene, container=item.spec, fill_ml=item.volume_ml * item.density
        )
        start = None if source is None else (source.x, source.y, source.surface_z)
        episode = run_episode(
            config, self._mjcf, ScriptedController(config), goal, seed=seed, start=start
        )
        return SkillResult(episode.success, episode.reason)
