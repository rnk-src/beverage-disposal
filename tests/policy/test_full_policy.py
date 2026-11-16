"""The full emptiness policy: what a classifier may and may not change, zone by zone."""

import dataclasses
import math
import zlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support.paths import ROOT
from test_policy import ALL_SIZES, BIG, BOTTLE, CONFIG, FAMILY, PET, POLICY, RULE, decide

from beverage_disposal.core.classifier import FeatureRanges, ModelBundle
from beverage_disposal.core.emptiness_eval import draw_case
from beverage_disposal.core.features import feature_names, feature_vector
from beverage_disposal.core.fullness import BIN, SET_ASIDE
from beverage_disposal.core.fullness_config import load_fullness_config
from beverage_disposal.core.np_calibration import min_samples, np_threshold
from beverage_disposal.core.pad import PadSensor, Weighing
from beverage_disposal.core.perception import exact_perception, load_perception_noise
from beverage_disposal.core.zones import Zone

PAD = load_fullness_config(ROOT / "configs" / "fullness.toml").pad
NOISE = load_perception_noise(ROOT / "configs" / "perception.toml")
NAMES = feature_names(FAMILY)
ANYTHING = FeatureRanges((-math.inf,) * len(NAMES), (math.inf,) * len(NAMES), (True,) * len(NAMES))
# No bottle weighs under a gram, so every bottle is outside these ranges.
NOTHING = dataclasses.replace(ANYTHING, hi=tuple(0.001 for _ in NAMES))
TAU = 0.9
AMBIGUOUS_ML = 20.0  # in a 2 L bottle the weight cannot tell 20 mL from far more


class Fixed:
    """A model that gives one score to every bottle and remembers what it was asked."""

    def __init__(self, score):
        self.value = score
        self.asked = []

    def score(self, x):
        self.asked.append(x)
        return self.value


class Raising:
    def score(self, x):
        raise RuntimeError("the model broke")


def bundled(model, tau=TAU, ranges=ANYTHING, policy=POLICY, taus=None):
    if not hasattr(model, "score"):
        model = Fixed(model)
    taus = dict.fromkeys(ALL_SIZES, tau) if taus is None else taus
    return dataclasses.replace(policy, bundle=ModelBundle(model, taus, ranges, NAMES, "sha", {}))


def summary(verdict):
    return verdict.action, verdict.decided_by, verdict.zone


# The surely-under zone: the classifier may only veto.


def test_a_surely_under_bottle_the_classifier_agrees_on_is_binned_by_the_rule():
    verdict = decide(policy=bundled(0.99))
    assert summary(verdict) == (BIN, "rule", "surely_under")
    assert verdict.score == 0.99 and verdict.bounds_ml is not None


def test_a_score_under_the_veto_score_sets_a_surely_under_bottle_aside():
    verdict = decide(policy=bundled(CONFIG.veto_score - 0.01))
    assert summary(verdict) == (SET_ASIDE, "classifier", "surely_under")
    assert verdict.score == CONFIG.veto_score - 0.01


def test_a_score_exactly_at_the_veto_score_does_not_veto():
    assert decide(policy=bundled(CONFIG.veto_score)).action == BIN


def test_the_veto_score_comes_from_the_config():
    strict = dataclasses.replace(POLICY, config=dataclasses.replace(CONFIG, veto_score=0.8))
    assert decide(policy=bundled(0.7, policy=strict)).action == SET_ASIDE
    assert decide(policy=bundled(0.7)).action == BIN


# The ambiguous zone: the classifier may bin, behind every gate.


def test_a_score_over_the_sizes_threshold_bins_an_ambiguous_bottle():
    verdict = decide(BIG, AMBIGUOUS_ML, policy=bundled(TAU + 0.01))
    assert summary(verdict) == (BIN, "classifier", "ambiguous")
    assert verdict.score == TAU + 0.01 and verdict.bounds_ml is not None


def test_a_score_exactly_at_the_threshold_does_not_bin():
    verdict = decide(BIG, AMBIGUOUS_ML, policy=bundled(TAU))
    assert summary(verdict) == (SET_ASIDE, "rule", "ambiguous")
    assert verdict.score == TAU


def test_the_threshold_is_that_of_the_bottles_own_size():
    taus = {**dict.fromkeys(ALL_SIZES, 0.1), "pet_2000ml": 0.95}
    assert decide(BIG, AMBIGUOUS_ML, policy=bundled(0.9, taus=taus)).action == SET_ASIDE
    taus = {**dict.fromkeys(ALL_SIZES, 0.95), "pet_2000ml": 0.1}
    assert decide(BIG, AMBIGUOUS_ML, policy=bundled(0.9, taus=taus)).action == BIN


def test_a_size_without_a_threshold_is_never_overturned():
    taus = {name: 0.0 for name in ALL_SIZES if name != "pet_2000ml"}
    verdict = decide(BIG, AMBIGUOUS_ML, policy=bundled(1.0, taus=taus))
    assert summary(verdict) == (SET_ASIDE, "rule", "ambiguous")


def test_an_infinite_threshold_is_never_beaten():
    assert decide(BIG, AMBIGUOUS_ML, policy=bundled(1.0, tau=math.inf)).action == SET_ASIDE


def test_without_allow_overturn_the_ambiguous_zone_is_never_binned():
    hardware = dataclasses.replace(POLICY, config=dataclasses.replace(CONFIG, allow_overturn=False))
    verdict = decide(BIG, AMBIGUOUS_ML, policy=bundled(1.0, tau=0.0, policy=hardware))
    assert summary(verdict) == (SET_ASIDE, "rule", "ambiguous")
    # The veto side of the classifier does not depend on the switch.
    assert decide(policy=bundled(0.0, policy=hardware)).decided_by == "classifier"


def two_class_case(policy):
    """A look that fits both 250 and 330 mL, at the lightest weight that is ambiguous."""
    seen = exact_perception(BOTTLE, 0.0)
    dims = dataclasses.replace(seen.dims, body_diameter=(0.056, 0.056), total_height=(0.15, 0.15))
    seen = dataclasses.replace(seen, dims=dims)
    for grams in range(5, 120):
        weighing = Weighing(mass_kg=grams / 1000, stable=True)
        found = policy.assess(seen, weighing)
        if found.zone is Zone.AMBIGUOUS:
            assert found.matches == ("pet_250ml", "pet_330ml")
            return seen, weighing
    raise AssertionError("no weight puts this look in the ambiguous zone")


def test_a_bottle_that_fits_two_sizes_is_not_overturned():
    policy = bundled(1.0, tau=0.0)
    verdict = policy.decide(*two_class_case(policy))
    assert summary(verdict) == (SET_ASIDE, "rule", "ambiguous")


# The surely-over zone: nothing overturns it.


def test_the_model_is_not_even_asked_about_a_surely_over_bottle():
    spy = Fixed(1.0)
    verdict = decide(volume_ml=250.0, policy=bundled(spy, tau=0.0, ranges=NOTHING))
    assert summary(verdict) == (SET_ASIDE, "rule", "surely_over")
    assert spy.asked == [] and verdict.score is None


# What the model is asked, and when.


def test_the_model_is_asked_once_with_the_feature_vector():
    spy = Fixed(0.99)
    policy = bundled(spy)
    seen, weighing = exact_perception(BOTTLE, 0.0), Weighing(BOTTLE.empty_mass_kg, True)
    policy.decide(seen, weighing)
    found = POLICY.assess(seen, weighing)
    expected = feature_vector(seen, weighing, found.bounds, found.v_bounds, FAMILY, PET)
    assert len(spy.asked) == 1
    np.testing.assert_array_equal(spy.asked[0], expected)


def test_the_model_is_not_asked_about_a_vetoed_bottle():
    spy = Fixed(1.0)
    verdict = decide(policy=bundled(spy), material="glass")
    assert verdict.decided_by == "veto:not_pet" and spy.asked == []


# Out of distribution, and a model that fails.

ZONE_CASES = [(BOTTLE, 0.0), (BIG, AMBIGUOUS_ML)]


@pytest.mark.parametrize(("bottle", "volume_ml"), ZONE_CASES, ids=["surely_under", "ambiguous"])
def test_a_bottle_outside_the_training_ranges_is_vetoed_and_the_model_not_asked(bottle, volume_ml):
    spy = Fixed(1.0)
    verdict = decide(bottle, volume_ml, policy=bundled(spy, tau=0.0, ranges=NOTHING))
    assert (verdict.action, verdict.decided_by) == (SET_ASIDE, "veto:out_of_distribution")
    assert spy.asked == []


@pytest.mark.parametrize("model", [Raising(), Fixed(math.nan)], ids=["raises", "nan"])
@pytest.mark.parametrize(("bottle", "volume_ml"), ZONE_CASES, ids=["surely_under", "ambiguous"])
def test_a_model_that_raises_or_scores_nan_sets_the_bottle_aside(model, bottle, volume_ml):
    verdict = decide(bottle, volume_ml, policy=bundled(model, tau=0.0))
    assert (verdict.action, verdict.decided_by) == (SET_ASIDE, "veto:model_error")


# Properties over drawn bottles and arbitrary classifiers.


def drawn(size, volume_ml, seed, crushed=False):
    rng = np.random.default_rng(seed)
    return draw_case(POLICY, PadSensor(PAD, rng), NOISE, size, volume_ml, rng, deformed=crushed)


CASES = st.builds(
    drawn,
    st.sampled_from(ALL_SIZES),
    st.floats(0.0, 80.0),
    st.integers(0, 2**32 - 1),
    st.booleans(),  # a crushed bottle is vetoed for low confidence
)
SCORES = st.one_of(st.floats(0.0, 1.0), st.just(math.nan))
POLICIES = st.builds(
    lambda score, tau, ranges, overturn: bundled(
        score,
        tau=tau,
        ranges=ranges,
        policy=dataclasses.replace(
            POLICY, config=dataclasses.replace(CONFIG, allow_overturn=overturn)
        ),
    ),
    SCORES,
    st.floats(0.0, 1.0),
    st.sampled_from([ANYTHING, NOTHING]),
    st.booleans(),
)


@settings(max_examples=300, derandomize=True, deadline=None)
@given(CASES, POLICIES)
def test_no_classifier_changes_a_veto_or_the_surely_over_zone(case, policy):
    alone = POLICY.decide(case.perception, case.weighing)
    full = policy.decide(case.perception, case.weighing)
    if alone.decided_by.startswith("veto:") or alone.zone == "surely_over":
        assert full == alone and full.action == SET_ASIDE


@settings(max_examples=300, derandomize=True, deadline=None)
@given(CASES, POLICIES)
def test_a_bin_in_the_ambiguous_zone_passed_every_gate(case, policy):
    full = policy.decide(case.perception, case.weighing)
    if full.zone == "ambiguous" and full.action == BIN:
        found = policy.assess(case.perception, case.weighing)
        assert full.decided_by == "classifier"
        assert policy.config.allow_overturn and policy.bundle.ranges is ANYTHING
        assert full.score > policy.bundle.tau_by_size[found.matches[0]]


@settings(max_examples=300, derandomize=True, deadline=None)
@given(CASES, POLICIES)
def test_in_the_surely_under_zone_a_classifier_can_only_turn_a_bin_into_a_set_aside(case, policy):
    alone = POLICY.decide(case.perception, case.weighing)
    full = policy.decide(case.perception, case.weighing)
    if alone.zone == "surely_under":
        assert full.action == SET_ASIDE or full.decided_by == "rule"
        assert full.decided_by != "classifier" or full.action == SET_ASIDE


def test_the_properties_meet_every_zone_and_a_veto():
    # Otherwise the three properties above could pass on nothing.
    met = set()
    for seed in range(60):
        case = drawn(ALL_SIZES[seed % len(ALL_SIZES)], (seed * 7) % 80, seed, seed % 4 == 0)
        verdict = POLICY.decide(case.perception, case.weighing)
        met.add(verdict.zone or "veto")
    assert met == {"surely_under", "ambiguous", "surely_over", "veto"}


# Adversarial models, each calibrated the way a real one is.


def ambiguous_cases(volume_ml, count, seed):
    """Drawn 2 L bottles of one volume that the weight cannot settle."""
    cases, rng = [], np.random.default_rng(seed)
    sensor = PadSensor(PAD, rng)
    while len(cases) < count:
        case = draw_case(POLICY, sensor, NOISE, "pet_2000ml", volume_ml, rng)
        if POLICY.assess(case.perception, case.weighing).zone is Zone.AMBIGUOUS:
            cases.append(case)
    return cases


class Hashed:
    """A score in [0, 1) that has nothing to do with how full the bottle is."""

    def score(self, x):
        return float(np.random.default_rng(zlib.crc32(x.tobytes())).random())


class Inverted:
    """Surer of empty the more the bottle weighs."""

    def score(self, x):
        return float(1.0 - math.exp(-x[NAMES.index("w_g")] / 100.0))


def calibrated_tau(model, cases):
    scores = [
        model.score(feature_vector(c.perception, c.weighing, f.bounds, f.v_bounds, FAMILY, PET))
        for c in cases
        for f in [POLICY.assess(c.perception, c.weighing)]
    ]
    return np_threshold(np.array(scores), CONFIG.alpha, CONFIG.delta)


@pytest.mark.medium
@pytest.mark.parametrize(
    ("model", "most_binned"),
    # A constant score never beats its own threshold. The other two get through about as often
    # as a fresh score beats the largest of 598, which alpha allows: under 3 in 400.
    [(Fixed(1.0), 0), (Fixed(0.0), 0), (Hashed(), 3), (Inverted(), 3)],
    ids=["always_empty", "always_full", "random", "inverted"],
)
def test_the_calibrated_threshold_stops_an_adversarial_model_at_35_ml(model, most_binned):
    needed = min_samples(CONFIG.alpha, CONFIG.delta)
    tau = calibrated_tau(model, ambiguous_cases(35.0, needed, seed=1))
    policy = bundled(model, tau=tau)
    fresh = ambiguous_cases(35.0, 400, seed=2)
    verdicts = [policy.decide(c.perception, c.weighing) for c in fresh]
    assert sum(v.action == BIN for v in verdicts) <= most_binned
    assert all(v.decided_by in ("rule", "classifier") for v in verdicts)


def test_the_rule_and_family_are_the_ones_the_file_names():
    assert POLICY.rule is RULE and POLICY.bundle is None
