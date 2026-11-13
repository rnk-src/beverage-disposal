"""The scripted controller in the real simulation: seeded spawns, three goals, and the baselines."""

import dataclasses

import numpy as np
import pytest
from support.paths import ROOT
from support.so101 import MJCF, SO101

from beverage_disposal.core.baselines import RandomController
from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.interfaces import PositionNoise
from beverage_disposal.core.layout import goals, sample_spawn
from beverage_disposal.core.scripted import ScriptedController
from beverage_disposal.sim.rollout import run_episode

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
GOALS = goals(CONFIG)
SEEDS = range(8)


def episode(controller, goal, seed, **options):
    return run_episode(CONFIG, SO101 / MJCF, controller, GOALS[goal], seed=seed, **options)


@pytest.mark.medium
@pytest.mark.parametrize("fill_ml", [0.0, 355.0], ids=["empty", "full"])
@pytest.mark.parametrize("goal", ["set_aside", "pad", "bin"])
def test_the_script_puts_the_can_on_each_goal_from_every_seeded_spawn(goal, fill_ml):
    config = dataclasses.replace(CONFIG, fill_ml=fill_ml)
    results = [
        run_episode(config, SO101 / MJCF, ScriptedController(config), GOALS[goal], seed=seed)
        for seed in SEEDS
    ]
    failed = [(seed, r.reason) for seed, r in zip(SEEDS, results, strict=True) if not r.success]
    assert not failed, failed


@pytest.mark.medium
def test_a_random_policy_puts_nothing_on_a_goal():
    results = [episode(RandomController(seed), "set_aside", seed) for seed in range(20)]
    assert sum(r.success for r in results) == 0


@pytest.mark.medium
def test_a_script_that_never_squeezes_puts_nothing_on_a_goal():
    results = [episode(ScriptedController(CONFIG, squeeze=False), "set_aside", s) for s in SEEDS]
    assert sum(r.success for r in results) == 0


@pytest.mark.medium
def test_the_script_still_works_with_two_millimetres_of_perception_noise():
    noise = PositionNoise(sigma=0.002)
    results = [episode(ScriptedController(CONFIG), "set_aside", s, noise=noise) for s in SEEDS]
    assert sum(r.success for r in results) >= 7


@pytest.mark.medium
def test_five_millimetres_of_perception_noise_breaks_the_unfiltered_script():
    # A measured limit, not a goal: the script reads the container position afresh every step and
    # waits for it within 3 to 5 mm, so jitter of that size keeps it from moving on (1 of 8 seeds
    # succeeded). A filter on the container estimate is the fix; this test then changes.
    noise = PositionNoise(sigma=0.005)
    results = [episode(ScriptedController(CONFIG), "set_aside", s, noise=noise) for s in SEEDS]
    assert sum(r.success for r in results) <= 3


@pytest.mark.medium
def test_each_seed_spawns_the_can_somewhere_else():
    # A script that never squeezes leaves the can where it started: the final position is the spawn.
    config = CONFIG
    stay = [episode(ScriptedController(config, squeeze=False), "set_aside", s) for s in (0, 1)]
    spawns = [sample_spawn(config, np.random.default_rng(s)) for s in (0, 1)]
    for result, spawn in zip(stay, spawns, strict=True):
        assert result.container[:2] == pytest.approx(spawn, abs=2e-3)
    assert not np.allclose(stay[0].container[:2], stay[1].container[:2], atol=1e-3)


@pytest.mark.medium
def test_the_same_seed_gives_the_same_episode():
    first = episode(ScriptedController(CONFIG), "set_aside", 3)
    second = episode(ScriptedController(CONFIG), "set_aside", 3)
    assert (first.success, first.steps) == (second.success, second.steps)
    assert first.container == pytest.approx(second.container, abs=1e-12)


def test_an_episode_that_runs_out_of_steps_fails_with_a_reason():
    result = episode(ScriptedController(CONFIG), "set_aside", 0, max_steps=5)
    assert not result.success
    assert "time" in result.reason
