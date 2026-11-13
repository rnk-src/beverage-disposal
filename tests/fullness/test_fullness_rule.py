"""The weight rule: bin only when sure the container holds under 30 mL, otherwise set aside."""

import dataclasses
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from support.paths import ROOT

from beverage_disposal.core.fullness import BIN, SET_ASIDE, bin_mask, decide
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.pad import Weighing
from beverage_disposal.core.registry import load_registry

CONFIGS = ROOT / "configs"
FULLNESS = load_fullness_config(CONFIGS / "fullness.toml")
RULE = FULLNESS.rule
REGISTRY = load_registry(CONFIGS / "containers.toml")
CAN = REGISTRY.lookup("can_12floz")  # lightest empty 13 g
BOTTLE = REGISTRY.lookup("bottle_500ml")  # lightest empty 18.5 g, cap must be confirmed
LIMIT_G = 26.5  # 30 mL * 0.95 g/mL - 2 g margin, worked by hand


def weighing(record, net_g, stable=True):
    return Weighing(mass_kg=record.lightest_empty_mass + net_g / 1000.0, stable=stable)


def verdict(net_g, record=CAN, **options):
    return decide(RULE, record, weighing(record, net_g, options.pop("stable", True)), **options)


def test_the_rule_file_says_thirty_millilitres_at_the_lightest_drink_less_a_margin():
    assert (RULE.threshold_ml, RULE.min_density_g_per_ml, RULE.margin_g) == (30.0, 0.95, 2.0)
    assert RULE.limit_g == pytest.approx(LIMIT_G)


def test_a_container_just_under_the_limit_is_binned_and_just_over_is_set_aside():
    assert verdict(26.49).action == BIN
    assert verdict(26.51).action == SET_ASIDE


@pytest.mark.parametrize("volume_ml", [29.9, 30.0, 30.1])
def test_thirty_millilitres_of_water_and_its_neighbours_are_set_aside(volume_ml):
    # Water weighs a gram per millilitre; the margin and the lightest-drink density keep these out.
    assert verdict(volume_ml).action == SET_ASIDE


@pytest.mark.parametrize("volume_ml", [0.0, 5.0, 25.0])
def test_an_empty_or_nearly_empty_container_is_binned(volume_ml):
    assert verdict(volume_ml).action == BIN


def test_a_reading_a_little_below_the_lightest_empty_weight_is_still_an_empty_container():
    assert verdict(-1.5).action == BIN  # inside the 2 g margin


def test_a_reading_well_below_the_lightest_empty_weight_is_not_trusted():
    result = verdict(-2.5)
    assert result.action == SET_ASIDE and "lighter" in result.reason


def test_a_container_of_an_unknown_type_is_set_aside():
    result = decide(RULE, None, Weighing(mass_kg=0.02, stable=True))
    assert result.action == SET_ASIDE and "unknown" in result.reason


def test_a_weighing_that_did_not_settle_is_set_aside():
    result = verdict(5.0, stable=False)
    assert result.action == SET_ASIDE and "settle" in result.reason


def test_a_bottle_needs_its_cap_confirmed_before_it_can_be_binned():
    assert verdict(5.0, BOTTLE).action == SET_ASIDE
    assert verdict(5.0, BOTTLE, cap_confirmed=False).action == SET_ASIDE
    capped = verdict(5.0, BOTTLE, cap_confirmed=True)
    assert capped.action == BIN
    assert "cap" in verdict(5.0, BOTTLE).reason


def test_a_can_needs_no_cap_check():
    assert verdict(5.0, CAN).action == BIN


def test_the_verdict_reports_the_net_weight_it_judged():
    assert verdict(12.3).net_g == pytest.approx(12.3, abs=1e-9)
    assert decide(RULE, None, Weighing(0.02, True)).net_g is None


@given(net=st.floats(-5.0, 60.0))
def test_the_vector_form_agrees_with_the_decision_for_every_weight(net):
    mask = bin_mask(np.array([net]), RULE)
    assert bool(mask[0]) == (verdict(net).action == BIN)


def test_the_margin_is_at_least_four_standard_deviations_of_one_weighing():
    pad = FULLNESS.pad
    sigma = math.sqrt(pad.noise_sigma_g**2 / pad.samples_averaged + pad.offset_sigma_g**2)
    assert RULE.margin_g >= 4 * sigma


def test_a_rule_without_a_margin_would_bin_a_container_the_measured_rule_keeps_out():
    # Proof the margin matters: 27 g is a gram over 0.95 * 30 - 2 but under 0.95 * 30.
    no_margin = dataclasses.replace(RULE, margin_g=0.0)
    assert decide(no_margin, CAN, weighing(CAN, 27.0)).action == BIN
    assert verdict(27.0).action == SET_ASIDE
