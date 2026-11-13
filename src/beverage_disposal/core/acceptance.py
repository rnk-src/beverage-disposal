"""Turns a list of sequencer runs into the headline numbers of the results file."""

from collections import defaultdict
from collections.abc import Sequence

from beverage_disposal.core.sequencer import BINNED, STOPPED, Run
from beverage_disposal.core.statistics import wilson_upper


def _key(volume_ml: float) -> str:
    return f"{volume_ml:g}"


def summarize(runs: Sequence[Run]) -> dict:
    """Counts and 95 % Wilson bounds. A stopped run counts as a failed pick-and-place."""
    if not runs:
        raise ValueError("no runs to summarize")
    threshold = runs[0].threshold_ml
    per_volume: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # n, misbinned
    for run in runs:
        if run.item.volume_ml >= threshold:
            counts = per_volume[_key(run.item.volume_ml)]
            counts[0] += 1
            counts[1] += int(run.misbinned)
    empties = [run for run in runs if run.item.volume_ml == 0.0]
    done = sum(run.destination != STOPPED for run in runs)
    return {
        "runs": len(runs),
        "threshold_ml": threshold,
        "binned_at_or_above_threshold": {
            volume: {"n": n, "k": k, "upper_95": wilson_upper(k, n)}
            for volume, (n, k) in sorted(per_volume.items(), key=lambda item: float(item[0]))
        },
        "empties_correctly_binned": {
            "n": len(empties),
            "k": sum(run.destination == BINNED for run in empties),
        },
        "pick_and_place": {
            "n": len(runs),
            "k": done,
            "lower_95": 1 - wilson_upper(len(runs) - done, len(runs)),
        },
    }
