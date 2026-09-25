"""Comparing two readouts of the same Gazebo scene."""

import pytest


def same_state(first, second):
    """Counts and field names must match exactly; times and positions use pytest.approx defaults.

    The defaults (relative 1e-6, absolute 1e-12) are the pytest project's own. MuJoCo's docs promise
    bit-exact repeats only for the same version on the same architecture, and contact amplifies any
    difference, so this is meant for contact-free scenes; contact scenes get physical tolerances.
    """
    if first.keys() != second.keys() or first["poses"].keys() != second["poses"].keys():
        return False
    others = first.keys() - {"iterations", "sim_time", "poses"}
    return (
        all(first[key] == second[key] for key in others)
        and first["iterations"] == second["iterations"]
        and first["sim_time"] == pytest.approx(second["sim_time"])
        and all(
            first["poses"][name] == pytest.approx(second["poses"][name]) for name in first["poses"]
        )
    )
