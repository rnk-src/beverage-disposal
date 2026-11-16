"""The safety tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

UNIT = "tests/safety/test_supervisor.py"
FAULTS_MODULE = "tests/safety/test_fault_injection.py"
SUPERVISOR = "core/supervisor.py"
SUPERVISED = "core/supervised.py"
SEQUENCER = "tests/sequencer/test_sequencer.py"


def widen_the_diameter_limit(package):
    edit(
        package,
        SUPERVISOR,
        "if grasp_width[1] > limits.max_container_diameter:",
        "if grasp_width[1] > 1.0:",
    )


def judge_the_width_by_its_low_end(package):
    edit(package, SUPERVISOR, "if grasp_width[1] > limits", "if grasp_width[0] > limits")


def forget_the_weight_limit(package):
    edit(
        package,
        SUPERVISOR,
        "if weighing.overloaded or weighing.mass_kg > limits.max_container_mass:",
        "if weighing.overloaded:",
    )


def trust_an_overloaded_pads_reading(package):
    edit(
        package,
        SUPERVISOR,
        "if weighing.overloaded or weighing.mass_kg > limits.max_container_mass:",
        "if weighing.mass_kg > limits.max_container_mass:",
    )


def report_a_set_aside_as_a_stop(package):
    edit(
        package,
        SUPERVISOR,
        "return self._trip(Outcome.SET_ASIDE, code, message)",
        "return self._trip(Outcome.SAFE_STOP, code, message)",
    )


def read_a_saturated_cell_as_no_overload(package):
    edit(
        package,
        "core/pad.py",
        "np.zeros(n, bool), np.ones(n, bool)",
        "np.zeros(n, bool), np.zeros(n, bool)",
    )


def lift_before_weighing(package):
    edit(
        package,
        "core/sequencer.py",
        "        if entry.outcome is Outcome.SAFE_STOP:",
        "        if False:",
    )


def leave_the_wet_outside_off_the_scale(package):
    edit(
        package,
        "core/sequencer.py",
        "item.volume_ml * item.density + item.wet_g",
        "item.volume_ml * item.density",
    )


def put_every_container_on_the_pads_centre(package):
    edit(package, "core/sequencer.py", "-INTAKE_SCATTER, INTAKE_SCATTER, 2", "0.0, 0.0, 2")


def bin_an_unknown_type(package):
    edit(
        package,
        "core/sequencer.py",
        "binned = verdict.action == BIN",
        "binned = verdict.action == BIN or record is None",
    )


def accept_a_failed_test_lift(package):
    edit(
        package,
        SUPERVISOR,
        "if gain_m < self._limits.min_test_lift_fraction * asked_m:",
        "if False:",
    )


def ignore_the_arm_force(package):
    edit(package, SUPERVISOR, "if arm_force_n > self._limits.max_arm_force:", "if False:")


def ignore_the_grip_torque(package):
    edit(package, SUPERVISOR, "if grip_torque_nm > self._limits.max_grip_torque:", "if False:")


def trust_every_observation(package):
    edit(
        package,
        SUPERVISOR,
        "if not all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in values):",
        "if False:",
    )


def forget_the_dropout(package):
    edit(package, SUPERVISOR, "        if observation is None:", "        if False:")


def never_time_out(package):
    edit(package, SUPERVISOR, "if seconds > self._limits.max_episode_seconds:", "if False:")


def let_a_moved_container_pass(package):
    edit(package, SUPERVISOR, "if moved > self._limits.container_position_tolerance:", "if False:")


def let_a_broken_check_escape(package):
    edit(
        package,
        SUPERVISOR,
        "        except Exception as error:",
        "        except KeyboardInterrupt as error:",
    )


def pass_a_non_finite_action_on(package):
    edit(package, SUPERVISOR, "        if not np.all(np.isfinite(action)):", "        if False:")


def stay_dark_when_stopping(package):
    edit(
        package,
        SUPERVISOR,
        '        self._indicators.sound("stop" if stop else "set_aside")',
        "        pass",
    )


def light_every_trip_red(package):
    edit(package, SUPERVISOR, '"red" if stop else "amber"', '"red"')


def never_check_the_supervised_observation(package):
    edit(
        package,
        SUPERVISED,
        "        decision = self._supervisor.check_observation(observation)",
        "        decision = self._supervisor.check_time(0.0)",
    )


STOP_GUARD = (
    "        if self.stop_event is not None:\n            return self._hold(observation)\n"
    "        self._steps += 1"
)


def keep_going_after_a_stop(package):
    edit(
        package,
        SUPERVISED,
        STOP_GUARD,
        "        self._steps += 1",
    )


FAULTS = [
    (
        UNIT,
        widen_the_diameter_limit,
        "test_a_grasp_that_may_be_wider_than_the_pads_can_take_is_refused[trip]",
    ),
    (
        UNIT,
        judge_the_width_by_its_low_end,
        "test_a_grasp_that_may_be_wider_than_the_pads_can_take_is_refused[trip]",
    ),
    (
        UNIT,
        forget_the_weight_limit,
        "test_a_container_weighed_over_the_limit_is_left_for_a_person_to_remove[trip]",
    ),
    (
        UNIT,
        trust_an_overloaded_pads_reading,
        "test_an_overloaded_pad_stops_even_though_its_reading_is_meaningless",
    ),
    (
        UNIT,
        report_a_set_aside_as_a_stop,
        "test_a_set_aside_is_reported_in_amber_and_is_not_a_stop",
    ),
    (
        "tests/fullness/test_pad_sensor.py",
        read_a_saturated_cell_as_no_overload,
        "test_a_load_over_the_cells_capacity_reads_overloaded_and_never_as_a_light_weight",
    ),
    (
        SEQUENCER,
        lift_before_weighing,
        "test_a_container_over_the_payload_is_never_handed_to_the_skill",
    ),
    (
        SEQUENCER,
        leave_the_wet_outside_off_the_scale,
        "test_water_on_the_outside_counts_in_the_weighed_mass",
    ),
    (
        SEQUENCER,
        put_every_container_on_the_pads_centre,
        "test_the_one_pick_is_from_the_pad_where_a_person_put_the_container",
    ),
    (
        SEQUENCER,
        bin_an_unknown_type,
        "test_an_empty_container_of_an_unknown_type_is_set_aside_not_binned",
    ),
    (
        UNIT,
        accept_a_failed_test_lift,
        "test_a_test_lift_that_does_not_raise_the_container_enough_stops_the_pick[trip]",
    ),
    (UNIT, ignore_the_arm_force, "test_an_arm_pressing_too_hard_is_stopped[trip]"),
    (UNIT, ignore_the_grip_torque, "test_a_squeeze_beyond_the_gripper_limit_is_stopped[trip]"),
    (UNIT, trust_every_observation, "test_an_observation_with_a_nan_or_an_infinity_is_not_trusted"),
    (UNIT, forget_the_dropout, "test_a_missing_observation_is_a_sensor_dropout_and_stops"),
    (UNIT, never_time_out, "test_an_episode_that_runs_too_long_is_stopped[trip]"),
    (
        UNIT,
        let_a_moved_container_pass,
        "test_a_container_that_is_not_where_it_was_seen_stops_the_pick",
    ),
    (
        UNIT,
        let_a_broken_check_escape,
        "test_a_check_that_raises_ends_in_a_safe_stop_and_never_escapes",
    ),
    (
        UNIT,
        pass_a_non_finite_action_on,
        "test_a_non_finite_action_is_replaced_by_standing_still_and_stops",
    ),
    (
        UNIT,
        stay_dark_when_stopping,
        "test_every_trip_is_reported_by_a_light_a_sound_and_a_message[wide]",
    ),
    (UNIT, light_every_trip_red, "test_a_stop_lights_red_and_a_set_aside_lights_amber"),
    (
        FAULTS_MODULE,
        never_check_the_supervised_observation,
        "test_nan_in_the_container_position_stops_the_arm_and_nothing_is_placed",
    ),
    (
        FAULTS_MODULE,
        keep_going_after_a_stop,
        "test_once_stopped_the_controller_ignores_a_good_observation_and_holds_still",
    ),
]


@pytest.mark.parametrize(
    ("module", "fault", "failing_test"),
    FAULTS,
    ids=lambda value: getattr(value, "__name__", None) if callable(value) else None,
)
def test_the_safety_tests_fail_on_a_mutated_package(tmp_path, module, fault, failing_test):
    result = run_on_mutated_package(tmp_path, module, fault)
    assert_the_test_failed(result, module, failing_test)
