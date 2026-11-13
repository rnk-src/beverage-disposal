"""A prescribed motion (no controller yet) closes the pads on the can and lifts it.

Rigid can, idealized servo: the can does not slosh and the gripper servo is the vendored position
actuator with a torque limit. The numbers show the grasp can work; they say nothing yet about a
real can or a real servo.
"""

import dataclasses

import pytest
from support.paths import ROOT
from support.so101 import MJCF, SO101

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.sim.grasp import prescribed_grasp

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
FULL_CAN_ML = 355.0
# The can may slide this far against the jaws during the lift and still count as held (assumed).
MAX_SLIP = 0.003


# The squeeze that suits each can (N m): a light can is thrown out by more than 0.3, a full one is
# dropped by less than 0.6 once the friction halves. The scene's default is the firm one, which
# the pick-and-place rollouts show works for both (their lift is slower and level).
GENTLE, FIRM = 0.3, 0.6


def grasp(config=CONFIG, **options):
    return prescribed_grasp(config, SO101 / MJCF, **options)


def suited(config):
    """`config` with the squeeze that suits its can: gentle when empty, firm when full."""
    force = FIRM if config.fill_ml > 0 else GENTLE
    return dataclasses.replace(config, grasp=dataclasses.replace(config.grasp, force_limit=force))


def scaled(config, factor):
    """Every friction coefficient in the scene multiplied by `factor`."""
    return dataclasses.replace(
        config,
        table=dataclasses.replace(config.table, friction=config.table.friction * factor),
        container=dataclasses.replace(
            config.container, friction=config.container.friction * factor
        ),
        gripper_pads=dataclasses.replace(
            config.gripper_pads, friction=config.gripper_pads.friction * factor
        ),
    )


@pytest.mark.parametrize("fill_ml", [0.0, FULL_CAN_ML], ids=["empty", "full"])
def test_the_can_is_lifted_with_the_gripper_and_does_not_slip(fill_ml):
    result = grasp(suited(dataclasses.replace(CONFIG, fill_ml=fill_ml)))
    assert result.lift_gain >= 0.9 * CONFIG.reach.lift_height
    assert result.slip < MAX_SLIP


@pytest.mark.medium
@pytest.mark.parametrize("factor", [0.5, 2.0])
@pytest.mark.parametrize("fill_ml", [0.0, FULL_CAN_ML], ids=["empty", "full"])
def test_the_grasp_holds_across_half_to_double_the_friction(factor, fill_ml):
    config = suited(scaled(dataclasses.replace(CONFIG, fill_ml=fill_ml), factor))
    result = grasp(config)
    assert result.lift_gain >= 0.9 * CONFIG.reach.lift_height
    assert result.slip < MAX_SLIP


@pytest.mark.medium
@pytest.mark.parametrize("fill_ml", [0.0, FULL_CAN_ML], ids=["empty", "full"])
def test_the_grasp_holds_at_half_the_physics_timestep(fill_ml):
    config = dataclasses.replace(
        CONFIG,
        fill_ml=fill_ml,
        physics=dataclasses.replace(CONFIG.physics, timestep=CONFIG.physics.timestep / 2),
    )
    result = grasp(suited(config))
    assert result.lift_gain >= 0.9 * CONFIG.reach.lift_height
    assert result.slip < MAX_SLIP


def contacts_scaled(scale, fill_ml):
    physics = dataclasses.replace(
        CONFIG.physics, contact_time_constant=CONFIG.physics.contact_time_constant * scale
    )
    return suited(dataclasses.replace(CONFIG, fill_ml=fill_ml, physics=physics))


@pytest.mark.medium
@pytest.mark.parametrize("fill_ml", [0.0, FULL_CAN_ML], ids=["empty", "full"])
def test_the_grasp_holds_with_stiffer_contacts(fill_ml):
    result = grasp(contacts_scaled(0.5, fill_ml))
    assert result.lift_gain >= 0.9 * CONFIG.reach.lift_height
    assert result.slip < MAX_SLIP


@pytest.mark.medium
def test_the_squeeze_is_a_trade_off_between_light_and_heavy_cans():
    # The limit is a real trade-off, not a bug. The gentle squeeze cannot hold a full can at half
    # the friction (it needs about 3.6 N per pad, 0.3 N m gives about 3 N), and the firm squeeze
    # throws the 14 g empty can out of the jaws.
    gentle = dataclasses.replace(
        CONFIG, grasp=dataclasses.replace(CONFIG.grasp, force_limit=GENTLE), fill_ml=FULL_CAN_ML
    )
    dropped = grasp(scaled(gentle, 0.5))
    assert dropped.lift_gain < 0.9 * CONFIG.reach.lift_height or dropped.slip >= MAX_SLIP
    firm = dataclasses.replace(
        CONFIG, grasp=dataclasses.replace(CONFIG.grasp, force_limit=FIRM), fill_ml=0.0
    )
    thrown = grasp(firm)
    assert thrown.lift_gain < 0.9 * CONFIG.reach.lift_height or thrown.slip >= MAX_SLIP


def test_the_same_grasp_twice_gives_the_same_numbers():
    first = grasp()
    second = grasp()
    assert (first.lift_gain, first.slip) == (second.lift_gain, second.slip)


@pytest.mark.medium
def test_slip_does_appear_when_the_pads_are_slippery():
    # Proof the test can fail: a tenth of the friction cannot hold a full can (mg = 3.6 N).
    result = grasp(suited(scaled(dataclasses.replace(CONFIG, fill_ml=FULL_CAN_ML), 0.1)))
    assert result.lift_gain < 0.9 * CONFIG.reach.lift_height or result.slip >= MAX_SLIP


def test_a_gripper_that_never_closes_does_not_lift_the_can():
    result = grasp(close=False)
    assert result.lift_gain < 0.01


@pytest.mark.medium
def test_a_stronger_squeeze_trades_the_full_can_at_half_friction_for_the_empty_can():
    # The limit is a real trade-off, not a bug: 0.6 N m holds the full can at half the friction
    # but throws the 14 g empty can out of the jaws.
    def with_force(config, newtons_metres):
        return dataclasses.replace(
            config, grasp=dataclasses.replace(config.grasp, force_limit=newtons_metres)
        )

    full_half = scaled(dataclasses.replace(CONFIG, fill_ml=FULL_CAN_ML), 0.5)
    strong = grasp(with_force(full_half, 0.6))
    assert strong.lift_gain >= 0.9 * CONFIG.reach.lift_height and strong.slip < MAX_SLIP
    empty = grasp(with_force(CONFIG, 0.6))
    assert empty.lift_gain < 0.9 * CONFIG.reach.lift_height or empty.slip >= MAX_SLIP
