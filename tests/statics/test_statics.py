"""The liquid wrench: what a rigid container's liquid does to a grasp when the container is tilted.

Model M (unvalidated): a rigid cylinder with a closed bottom and an open top, liquid at rest with a
horizontal free surface, no sloshing, no surface tension. The tilt is about a horizontal axis.
"""

import math

import pytest

from beverage_disposal.core.statics import liquid_wrench, worst_case_torque_error

RADIUS = 0.033  # the 12 fl oz can: 66 mm across, 122 mm high, grasped 100 mm up
HEIGHT = 0.122
GRASP = 0.10
FULL_ML = math.pi * RADIUS**2 * HEIGHT * 1e6  # about 417 mL to the brim
G = 9.81


def wrench(fill_ml, tilt_deg, density=1.0, radius=RADIUS, height=HEIGHT, grasp=GRASP):
    return liquid_wrench(radius, height, grasp, fill_ml, math.radians(tilt_deg), density)


def test_an_upright_container_puts_no_torque_on_the_grasp_whatever_it_holds():
    for fill in (0.0, 30.0, 200.0, 400.0):
        result = wrench(fill, 0.0)
        assert result.torque_nm == pytest.approx(0.0, abs=1e-9)
        assert result.retained_ml == pytest.approx(fill, rel=1e-6)
        assert result.spilled_ml == 0.0


def test_an_empty_container_puts_no_liquid_torque_at_any_tilt():
    assert wrench(0.0, 45.0).torque_nm == 0.0


def test_the_liquid_that_stays_and_the_liquid_that_spills_add_up_to_what_was_poured():
    for fill, tilt in [(50.0, 30.0), (200.0, 50.0), (300.0, 40.0), (416.0, 20.0)]:
        result = wrench(fill, tilt)
        assert result.retained_ml + result.spilled_ml == pytest.approx(fill, rel=1e-6)


def test_a_tilted_container_matches_the_closed_form_for_a_level_that_misses_both_ends():
    # Liquid depth d at the axis, tilt t: the lever of its centroid about the grasp axis is
    # -tan(t) R^2 / (4 d) * cos(t) - (d/2 + tan(t)^2 R^2 / (8 d) - grasp) * sin(t), derived by hand.
    # Valid while the level misses the bottom and the rim: d >= R tan(t) and d + R tan(t) <= H.
    fill, tilt = 150.0, 0.5
    d = fill * 1e-6 / (math.pi * RADIUS**2)
    lever = -math.tan(tilt) * RADIUS**2 / (4 * d) * math.cos(tilt)
    lever -= (d / 2 + math.tan(tilt) ** 2 * RADIUS**2 / (8 * d) - GRASP) * math.sin(tilt)
    expected = fill * 1e-3 * G * lever
    assert liquid_wrench(RADIUS, HEIGHT, GRASP, fill, tilt, 1.0).torque_nm == pytest.approx(
        expected, rel=2e-3
    )


def test_a_full_container_at_the_angle_that_cuts_it_in_half_keeps_exactly_half():
    # The plane through the low rim point and the high bottom point splits a cylinder in two equal
    # parts when tan(t) = H / (2 R); a brim-full container then keeps half.
    tilt = math.degrees(math.atan(HEIGHT / (2 * RADIUS)))
    result = wrench(FULL_ML, tilt)
    assert result.retained_ml == pytest.approx(FULL_ML / 2, rel=1e-3)
    assert result.spilled_ml == pytest.approx(FULL_ML / 2, rel=1e-3)


def test_a_half_full_container_starts_to_spill_at_the_same_cutting_angle():
    critical = math.degrees(math.atan(HEIGHT / (2 * RADIUS)))  # about 61.6 degrees
    assert wrench(FULL_ML / 2, critical - 5.0).spilled_ml == 0.0
    assert wrench(FULL_ML / 2, critical + 5.0).spilled_ml > 1.0


def test_the_liquid_torque_scales_with_the_density_and_nothing_else():
    assert wrench(100.0, 40.0, density=1.05).torque_nm == pytest.approx(
        1.05 * wrench(100.0, 40.0, density=1.0).torque_nm, rel=1e-9
    )


def test_the_torque_grows_with_the_volume_at_the_working_tilt():
    # 60 degrees is the tilt the routine uses (2.6c); between 5 and 120 mL more liquid pulls harder.
    torques = [abs(wrench(v, 60.0).torque_nm) for v in (5.0, 15.0, 30.0, 60.0, 120.0)]
    assert torques == sorted(torques)
    assert torques[0] < torques[2] < torques[4]


def test_the_torque_flips_sign_with_the_direction_of_the_tilt():
    assert wrench(100.0, -40.0).torque_nm == pytest.approx(-wrench(100.0, 40.0).torque_nm, rel=1e-6)


def test_a_tilt_of_ninety_degrees_or_more_is_refused():
    with pytest.raises(ValueError, match="tilt"):
        wrench(100.0, 90.0)


def test_a_negative_fill_is_refused():
    with pytest.raises(ValueError, match="fill"):
        wrench(-1.0, 30.0)


def test_with_no_uncertainty_the_worst_case_error_is_zero():
    zero = worst_case_torque_error(
        RADIUS,
        HEIGHT,
        GRASP,
        30.0,
        math.radians(60.0),
        density=1.0,
        radius_error=0.0,
        height_error=0.0,
        grasp_error=0.0,
        density_error=0.0,
        tilt_error=0.0,
    )
    assert zero == 0.0


def test_the_worst_case_error_is_at_least_as_big_as_any_one_parameter_alone():
    kwargs = dict(
        radius_error=0.001,
        height_error=0.002,
        grasp_error=0.005,
        density_error=0.05,
        tilt_error=math.radians(1.0),
    )
    everything = worst_case_torque_error(
        RADIUS, HEIGHT, GRASP, 30.0, math.radians(60.0), 1.0, **kwargs
    )
    for name in kwargs:
        alone = {key: (value if key == name else 0.0) for key, value in kwargs.items()}
        one = worst_case_torque_error(RADIUS, HEIGHT, GRASP, 30.0, math.radians(60.0), 1.0, **alone)
        assert everything >= one - 1e-12, name


def test_the_density_error_alone_moves_the_torque_by_that_fraction_of_it():
    nominal = abs(wrench(30.0, 60.0).torque_nm)
    only_density = worst_case_torque_error(
        RADIUS,
        HEIGHT,
        GRASP,
        30.0,
        math.radians(60.0),
        1.0,
        radius_error=0.0,
        height_error=0.0,
        grasp_error=0.0,
        density_error=0.05,
        tilt_error=0.0,
    )
    assert only_density == pytest.approx(0.05 * nominal, rel=1e-9)


def test_the_recorded_sensitivity_at_thirty_millilitres_and_sixty_degrees():
    # Measured 2026-11-13: 18.7 mN m nominal; the five errors together move it by at most 2.6 mN m
    # (grasp height 1.3, density 0.9, tilt 0.2, radius 0.1, height 0). That is about 4 mL of liquid.
    nominal = abs(wrench(30.0, 60.0).torque_nm)
    assert nominal == pytest.approx(0.01866, abs=2e-4)
    worst = worst_case_torque_error(
        RADIUS,
        HEIGHT,
        GRASP,
        30.0,
        math.radians(60.0),
        1.0,
        radius_error=0.001,
        height_error=0.002,
        grasp_error=0.005,
        density_error=0.05,
        tilt_error=math.radians(1.0),
    )
    assert worst == pytest.approx(0.00261, abs=2e-4)
