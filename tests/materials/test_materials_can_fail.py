"""The materials tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

TESTS = "tests/materials/test_materials.py"
MATERIALS = "core/materials.py"


def skip_the_range_check(package):
    edit(package, MATERIALS, "        if self.lo > self.hi:", "        if False:")


def name_every_material_the_same(package):
    edit(package, MATERIALS, '{"name": name, **table}', '{"name": "material", **table}')


FAULTS = [
    (
        skip_the_range_check,
        "test_a_range_with_its_low_end_above_its_high_end_is_refused",
    ),
    (
        name_every_material_the_same,
        "test_the_file_holds_pet_named_after_its_table",
    ),
]


@pytest.mark.parametrize(
    ("fault", "failing_test"), FAULTS, ids=lambda value: getattr(value, "__name__", None)
)
def test_the_materials_tests_fail_on_a_mutated_package(tmp_path, fault, failing_test):
    result = run_on_mutated_package(tmp_path, TESTS, fault)
    assert_the_test_failed(result, TESTS, failing_test)
