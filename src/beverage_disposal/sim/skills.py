"""The pick-and-place skill as the simulation runs it: the scripted controller in MuJoCo."""

import dataclasses

from beverage_disposal.core.bottle_family import BottleFamily, scene_for_bottle
from beverage_disposal.core.config import SceneConfig
from beverage_disposal.core.layout import Goal
from beverage_disposal.core.scripted import ScriptedController
from beverage_disposal.core.sequencer import Item, Placement, SkillResult
from beverage_disposal.sim.rollout import run_episode


class SimSkill:
    """Runs one leg as a full simulated episode. Each leg starts with the arm at its ready pose."""

    def __init__(self, scene: SceneConfig, mjcf, on_step=None, family: BottleFamily | None = None):
        self._scene = scene
        self._mjcf = mjcf
        self._on_step = on_step
        self._family = family

    def scene_for(self, item: Item) -> SceneConfig:
        """The scene with this item as its container: a drawn bottle, or the item's cylinder."""
        if item.bottle is None:
            # The cylinder's liquid weighs a gram per millilitre: a denser drink counts as more.
            return dataclasses.replace(
                self._scene, container=item.spec, fill_ml=item.volume_ml * item.density
            )
        if self._family is None:
            raise ValueError("a bottle item needs the bottle family for its grasp")
        size = self._family.classes[item.bottle.size]
        return scene_for_bottle(self._scene, item.bottle, size, item.volume_ml, item.density)

    def pick_and_place(
        self, source: Placement | None, goal: Goal, *, item: Item, seed: int
    ) -> SkillResult:
        config = self.scene_for(item)
        start = None if source is None else (source.x, source.y, source.surface_z)
        episode = run_episode(
            config,
            self._mjcf,
            ScriptedController(config),
            goal,
            seed=seed,
            start=start,
            on_step=self._on_step,
        )
        return SkillResult(episode.success, episode.reason)
