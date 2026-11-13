"""The results summary: counts per volume, Wilson bounds, and the headline numbers."""

import pytest
from test_sequencer import CAN, item

from beverage_disposal.core.acceptance import summarize
from beverage_disposal.core.sequencer import BINNED, SET_ASIDE_AREA, STOPPED, Run
from beverage_disposal.core.statistics import wilson_upper


def run(volume_ml, destination):
    return Run(item(CAN, volume_ml), None, destination, [], 30.0)


def test_misbins_are_counted_per_volume_and_only_from_thirty_millilitres_up():
    runs = [run(35.0, SET_ASIDE_AREA)] * 9 + [run(35.0, BINNED)] + [run(10.0, BINNED)] * 5
    summary = summarize(runs)
    assert summary["binned_at_or_above_threshold"]["35"] == {
        "n": 10,
        "k": 1,
        "upper_95": pytest.approx(wilson_upper(1, 10)),
    }
    assert "10" not in summary["binned_at_or_above_threshold"]


def test_empties_correctly_binned_counts_empties_that_reached_the_bin():
    runs = [run(0.0, BINNED)] * 3 + [run(0.0, SET_ASIDE_AREA), run(0.0, STOPPED)]
    assert summarize(runs)["empties_correctly_binned"] == {"n": 5, "k": 3}


def test_a_stopped_run_is_a_failed_pick_and_place_and_lowers_the_success_bound():
    good = [run(0.0, BINNED)] * 19
    assert summarize(good + [run(0.0, STOPPED)])["pick_and_place"]["k"] == 19
    assert (
        summarize(good + [run(0.0, STOPPED)])["pick_and_place"]["lower_95"]
        < summarize(good)["pick_and_place"]["lower_95"]
    )


def test_the_results_file_is_stable_json_data():
    import json

    json.dumps(summarize([run(0.0, BINNED)]))
