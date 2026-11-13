"""The registry tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

REGISTRY_TESTS = "tests/containers/test_registry.py"
SCENE_TESTS = "tests/scene/test_scene_config.py"
REGISTRY = "core/registry.py"
CONTAINERS = "core/containers.py"
CONFIG = "core/config.py"


def keep_every_request(package):
    edit(
        package,
        REGISTRY,
        "others = tuple(r for r in self.pending if r.type_id != type_id)",
        "others = self.pending",
    )


IN_PLACE = (
    "        object.__setattr__(self, 'pending', (*others, request))\n        return None, self"
)


def change_the_registry_in_place(package):
    edit(
        package,
        REGISTRY,
        "        return None, replace(self, pending=(*others, request))",
        IN_PLACE,
    )


def forget_to_clear_the_request(package):
    edit(
        package,
        REGISTRY,
        "pending = tuple(r for r in self.pending if r.type_id != record.type_id)",
        "pending = self.pending",
    )


def allow_a_duplicate_type(package):
    edit(package, REGISTRY, "        if record.type_id in self.records:", "        if False:")


def skip_the_weight_check(package):
    edit(
        package,
        CONTAINERS,
        "        if not 0 < self.lightest_empty_mass <= self.spec.empty_mass:",
        "        if False:",
    )


def skip_the_name_check(package):
    edit(package, CONTAINERS, "        if self.spec.name != self.type_id:", "        if False:")


def swallow_an_unknown_container_type(package):
    edit(package, CONFIG, "    if record is None:", "    if False:")


def read_the_wrong_registry_file(package):
    edit(package, CONFIG, 'path.parent / "containers.toml"', 'path.parent / "scene.toml"')


FAULTS = [
    (
        REGISTRY_TESTS,
        keep_every_request,
        "test_meeting_the_same_unknown_type_again_keeps_one_request_with_the_latest_weight",
    ),
    (
        REGISTRY_TESTS,
        change_the_registry_in_place,
        "test_an_unknown_type_is_not_found_and_leaves_a_request_for_the_owner",
    ),
    (
        REGISTRY_TESTS,
        forget_to_clear_the_request,
        "test_enrolling_adds_the_record_and_clears_its_request",
    ),
    (
        REGISTRY_TESTS,
        allow_a_duplicate_type,
        "test_enrolling_a_type_that_already_exists_is_refused",
    ),
    (
        REGISTRY_TESTS,
        skip_the_weight_check,
        "test_a_record_whose_lightest_weight_is_above_its_typical_weight_is_refused",
    ),
    (REGISTRY_TESTS, skip_the_name_check, "test_a_record_whose_spec_has_another_name_is_refused"),
    (
        REGISTRY_TESTS,
        swallow_an_unknown_container_type,
        "test_a_scene_naming_an_unknown_container_type_is_rejected_by_name",
    ),
    (
        SCENE_TESTS,
        read_the_wrong_registry_file,
        "test_the_scene_names_the_can_and_gets_its_figures_from_the_registry",
    ),
]


@pytest.mark.parametrize(
    ("module", "fault", "failing_test"),
    FAULTS,
    ids=lambda value: getattr(value, "__name__", None) if callable(value) else None,
)
def test_the_registry_tests_fail_on_a_mutated_package(tmp_path, module, fault, failing_test):
    result = run_on_mutated_package(tmp_path, module, fault)
    assert_the_test_failed(result, module, failing_test)
