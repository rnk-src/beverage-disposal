"""The bounds, zones, policy and grid tests are run against mutated copies and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

BOUNDS_TESTS = "tests/bounds/test_bounds.py"
ZONE_TESTS = "tests/zones/test_zones.py"
POLICY_TESTS = "tests/policy/test_policy.py"
EVAL_TESTS = "tests/policy/test_emptiness_eval.py"
PERCEPTION_TESTS = "tests/perception/test_perception.py"
SEQUENCER_TESTS = "tests/sequencer/test_sequencer.py"
FEATURE_TESTS = "tests/features/test_features.py"
DATASET_TESTS = "tests/train/test_dataset.py"
DATASET = "train/emptiness/dataset.py"
BOUNDS = "core/bounds.py"
ZONES = "core/zones.py"
POLICY = "core/policy.py"
EVAL = "core/emptiness_eval.py"
HAND = "test_a_500_ml_bottle_worked_by_hand"
BRACKET = "test_every_drawn_bottle_weighs_between_its_bounds_dry_or_wet[pet_500ml]"
GRID = "test_no_bottle_of_the_grid_is_put_in_a_zone_its_true_volume_contradicts"


def use_the_thick_wall_for_the_lightest_bottle(package):
    edit(
        package,
        BOUNDS,
        "shell_kg(0, m.density_g_cm3.lo, m.t_side_mm.lo, m.t_base_mm.lo)",
        "shell_kg(0, m.density_g_cm3.lo, m.t_side_mm.hi, m.t_base_mm.lo)",
    )


def use_the_dense_plastic_for_the_lightest_bottle(package):
    edit(package, BOUNDS, "shell_kg(0, m.density_g_cm3.lo,", "shell_kg(0, m.density_g_cm3.hi,")


def leave_the_shoulder_out_of_the_shell(package):
    edit(package, BOUNDS, "(side[end] + shoulder[end]) * t_side_mm", "side[end] * t_side_mm")


def leave_the_cap_out_of_the_heaviest_bottle(package):
    edit(
        package, BOUNDS, "m.finish_g.hi + m.cap_g.hi + m.label_g.hi", "m.finish_g.hi + m.label_g.hi"
    )


def leave_the_wet_outside_out_of_the_heaviest_bottle(package):
    edit(package, BOUNDS, " + m.label_g.hi + m.wet_g.hi", " + m.label_g.hi")


def plug_the_small_radius_into_the_shoulders_low_end(package):
    edit(
        package,
        BOUNDS,
        "math.pi * (big[0] ** 2 - small[1] ** 2),",
        "math.pi * (big[0] + small[0]) * math.hypot(height[0], big[0] - small[0]),",
    )


def take_the_full_height_as_the_body(package):
    edit(
        package,
        BOUNDS,
        "body_hi = dims.total_height[1] - dims.shoulder_height[0] - dims.neck_height[0]",
        "body_hi = dims.total_height[1]",
    )


def call_exactly_thirty_surely_under(package):
    edit(package, ZONES, "if v_hi < threshold_ml:", "if v_hi <= threshold_ml:")


def call_just_under_thirty_surely_over(package):
    edit(package, ZONES, "if v_lo >= threshold_ml:", "if v_lo >= threshold_ml - 0.01:")


def drop_the_scales_margin(package):
    edit(
        package,
        ZONES,
        "+ rule.margin_g) / rule.min_density_g_per_ml",
        ") / rule.min_density_g_per_ml",
    )


def swap_the_drink_densities(package):
    edit(
        package,
        ZONES,
        "+ rule.margin_g) / rule.min_density_g_per_ml",
        "+ rule.margin_g) / rule.max_density_g_per_ml",
    )


def let_the_least_liquid_go_negative(package):
    edit(package, ZONES, "return max(least, 0.0), most", "return least, most")


def skip_the_material_check(package):
    edit(package, POLICY, "if perception.material != PET:", "if False:")


def skip_the_settled_check(package):
    edit(package, POLICY, "if not weighing.stable:", "if False:")


def veto_a_confidence_at_the_minimum(package):
    edit(
        package,
        POLICY,
        "perception.confidence < self.config.min_confidence",
        "perception.confidence <= self.config.min_confidence",
    )


def bin_the_ambiguous_zone(package):
    edit(
        package,
        POLICY,
        'return verdict(SET_ASIDE, "the weight cannot tell")',
        'return verdict(BIN, "the weight cannot tell")',
    )


def accept_any_matching_size(package):
    edit(
        package,
        POLICY,
        "if not matches or not set(matches) <= set(self.supported()):",
        "if not matches or not set(matches) & set(self.supported()):",
    )


def bin_a_weight_lighter_than_any_bottle(package):
    edit(package, POLICY, "if v_bounds[1] < 0:", "if False:")


def let_the_weight_contradict_the_level(package):
    edit(
        package,
        POLICY,
        "if level_ml is not None and level_ml[1] + self.config.level_slack_ml < v_bounds[0]:",
        "if False:",
    )


def ignore_the_level_slack(package):
    edit(
        package,
        POLICY,
        "level_ml[1] + self.config.level_slack_ml < v_bounds[0]",
        "level_ml[1] < v_bounds[0]",
    )


def compare_the_levels_low_end(package):
    edit(
        package,
        POLICY,
        "level_ml[1] + self.config.level_slack_ml",
        "level_ml[0] + self.config.level_slack_ml",
    )


def copy_the_weight_in_kilograms(package):
    edit(package, "core/features.py", "weighing.mass_kg * KG_TO_G,", "weighing.mass_kg,")


def call_an_unseen_level_zero(package):
    edit(
        package, "core/features.py", "*(level or (math.nan, math.nan)),", "*(level or (0.0, 0.0)),"
    )


def flag_only_the_first_matching_size(package):
    edit(package, "core/features.py", "float(name in matches)", "float(name == matches[0])")


def leave_the_foreign_object_off_the_scale(package):
    edit(
        package, EVAL, "(volume_ml * density + wet_g + foreign_g)", "(volume_ml * density + wet_g)"
    )


def see_a_crushed_bottle_as_sound(package):
    edit(
        package,
        EVAL,
        "perceive(bottle, volume_ml, noise, rng, deformed=deformed)",
        "perceive(bottle, volume_ml, noise, rng)",
    )


def label_exactly_thirty_as_under(package):
    edit(
        package,
        DATASET,
        "case.volume_ml < policy.rule.threshold_ml,",
        "case.volume_ml <= 30.0 + 1.0,",
    )


def draw_every_split_from_one_stream(package):
    edit(package, DATASET, "return (seed, SPLIT_STREAMS[split])", "return (seed, 0)")


def fill_the_cells_from_every_zone(package):
    edit(
        package,
        DATASET,
        "kept += rows.add(case, size, only_zone=Zone.AMBIGUOUS)",
        "kept += rows.add(case, size)",
    )


def end_every_last_cell_at_the_nominal_size(package):
    edit(
        package,
        DATASET,
        "top = min(size_nominal_ml, config.payload_kg * KG_TO_ML_OF_WATER)",
        "top = size_nominal_ml",
    )


def act_on_every_size_by_default(package):
    edit(
        package,
        POLICY,
        "return self.family.supported() if self.sizes is None else self.sizes",
        "return tuple(self.family.classes) if self.sizes is None else self.sizes",
    )


def weigh_every_drink_as_water(package):
    edit(package, EVAL, "(volume_ml * density + wet_g", "(volume_ml + wet_g")


def weigh_every_bottle_dry(package):
    edit(
        package,
        EVAL,
        "(volume_ml * density + wet_g + foreign_g)",
        "(volume_ml * density + foreign_g)",
    )


def use_one_stream_for_every_seed(package):
    edit(package, EVAL, "default_rng([seed, ", "default_rng([0, ")


def match_a_class_by_width_alone(package):
    edit(
        package,
        "core/bottle_family.py",
        "if overlap(dims.body_diameter, width.lo, width.hi) and overlap(",
        "if overlap(dims.body_diameter, width.lo, width.hi) or not overlap(",
    )


def send_a_judged_pet_bottle_for_enrolment(package):
    edit(
        package,
        "core/sequencer.py",
        "            return verdict\n        record, self.registry",
        "        record, self.registry",
    )


FAULTS = [
    (use_the_thick_wall_for_the_lightest_bottle, BOUNDS_TESTS, HAND),
    (use_the_dense_plastic_for_the_lightest_bottle, BOUNDS_TESTS, HAND),
    (leave_the_shoulder_out_of_the_shell, BOUNDS_TESTS, BRACKET),
    (leave_the_cap_out_of_the_heaviest_bottle, BOUNDS_TESTS, BRACKET),
    (leave_the_wet_outside_out_of_the_heaviest_bottle, BOUNDS_TESTS, BRACKET),
    (
        plug_the_small_radius_into_the_shoulders_low_end,
        BOUNDS_TESTS,
        "test_the_shoulder_bounds_bracket_the_true_area_of_random_shapes",
    ),
    (
        take_the_full_height_as_the_body,
        BOUNDS_TESTS,
        "test_a_height_interval_that_leaves_no_room_for_a_body_is_refused",
    ),
    (call_exactly_thirty_surely_under, ZONE_TESTS, "test_the_zone_edges_at_exactly_thirty"),
    (call_just_under_thirty_surely_over, ZONE_TESTS, "test_the_zone_edges_at_exactly_thirty"),
    (drop_the_scales_margin, ZONE_TESTS, GRID),
    (swap_the_drink_densities, ZONE_TESTS, GRID),
    (let_the_least_liquid_go_negative, ZONE_TESTS, "test_the_least_liquid_is_never_negative"),
    (
        skip_the_material_check,
        POLICY_TESTS,
        "test_a_veto_stands_even_when_the_weight_says_surely_under",
    ),
    (
        skip_the_settled_check,
        POLICY_TESTS,
        "test_each_veto_alone_sets_an_otherwise_empty_bottle_aside_under_its_own_code"
        "[unsettled-kwargs1]",
    ),
    (
        veto_a_confidence_at_the_minimum,
        POLICY_TESTS,
        "test_a_confidence_exactly_at_the_minimum_is_acted_on",
    ),
    (
        bin_the_ambiguous_zone,
        POLICY_TESTS,
        "test_a_bottle_in_the_ambiguous_zone_is_set_aside_without_a_classifier",
    ),
    (
        accept_any_matching_size,
        POLICY_TESTS,
        "test_a_bottle_that_could_also_be_an_unsupported_size_is_vetoed",
    ),
    (
        bin_a_weight_lighter_than_any_bottle,
        POLICY_TESTS,
        "test_each_veto_alone_sets_an_otherwise_empty_bottle_aside_under_its_own_code"
        "[lighter_than_possible-kwargs3]",
    ),
    (
        act_on_every_size_by_default,
        POLICY_TESTS,
        "test_without_named_sizes_the_policy_acts_only_on_sizes_that_passed_the_gate",
    ),
    (
        let_the_weight_contradict_the_level,
        POLICY_TESTS,
        "test_a_weight_that_proves_far_more_liquid_than_the_level_shows_is_vetoed",
    ),
    (
        ignore_the_level_slack,
        POLICY_TESTS,
        "test_the_slack_is_how_much_more_the_weight_may_show_before_the_veto",
    ),
    (
        compare_the_levels_low_end,
        POLICY_TESTS,
        "test_a_level_interval_that_may_reach_the_weights_liquid_is_not_a_contradiction",
    ),
    (
        copy_the_weight_in_kilograms,
        FEATURE_TESTS,
        "test_sizes_weight_bounds_and_confidence_are_copied_in_their_units",
    ),
    (call_an_unseen_level_zero, FEATURE_TESTS, "test_an_unseen_level_is_missing_not_zero"),
    (
        flag_only_the_first_matching_size,
        FEATURE_TESTS,
        "test_the_size_flags_mark_every_class_the_measured_size_could_be",
    ),
    (
        leave_the_foreign_object_off_the_scale,
        EVAL_TESTS,
        "test_a_foreign_object_is_on_the_scale_but_the_level_stays_that_of_the_liquid",
    ),
    (
        see_a_crushed_bottle_as_sound,
        EVAL_TESTS,
        "test_a_crushed_bottle_is_seen_as_one_and_needs_the_noise_model",
    ),
    (
        label_exactly_thirty_as_under,
        DATASET_TESTS,
        "test_the_label_is_whether_the_true_volume_is_under_thirty",
    ),
    (draw_every_split_from_one_stream, DATASET_TESTS, "test_the_splits_share_no_bottle"),
    (
        fill_the_cells_from_every_zone,
        DATASET_TESTS,
        "test_every_filled_calibration_cell_has_exactly_the_rows_asked_all_ambiguous_and_in_its_bin",
    ),
    (
        end_every_last_cell_at_the_nominal_size,
        DATASET_TESTS,
        "test_the_cells_of_a_size_end_at_what_it_can_hold_or_the_arm_can_carry",
    ),
    (weigh_every_drink_as_water, EVAL_TESTS, "test_the_drink_is_weighed_at_its_density"),
    (weigh_every_bottle_dry, EVAL_TESTS, "test_water_on_the_outside_is_on_the_scale"),
    (
        use_one_stream_for_every_seed,
        EVAL_TESTS,
        "test_the_same_seed_gives_the_same_cells_and_another_seed_other_bottles",
    ),
    (
        match_a_class_by_width_alone,
        PERCEPTION_TESTS,
        "test_a_bottle_too_tall_for_its_width_matches_no_class",
    ),
    (
        send_a_judged_pet_bottle_for_enrolment,
        SEQUENCER_TESTS,
        "test_a_pet_bottle_judged_by_the_policy_is_not_sent_for_enrolment",
    ),
]


@pytest.mark.parametrize(
    ("fault", "tests", "failing_test"), FAULTS, ids=lambda value: getattr(value, "__name__", None)
)
def test_the_policy_tests_fail_on_a_mutated_package(tmp_path, fault, tests, failing_test):
    result = run_on_mutated_package(tmp_path, tests, fault)
    assert_the_test_failed(result, tests, failing_test)
