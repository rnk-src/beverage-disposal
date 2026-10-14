"""Step-response metrics from a time series, so a MuJoCo log and a Gazebo log are measured alike."""

import numpy as np

# Rise time is the 10 % to 90 % crossing; settling is the last time outside a 2 % band.
RISE_LOW, RISE_HIGH = 0.1, 0.9
SETTLE_BAND = 0.02
# The steady-state value is the mean of the last tenth of the record.
TAIL_FRACTION = 0.1


def step_response_metrics(times, values, start, target):
    """Rise time (s), overshoot (% of the step), settling time (s) and steady-state error.

    `start` and `target` are the signal before and after the step; the time axis starts at the step.
    The steady-state error is the size of the miss in the signal's own units (a joint's sag, say).
    """
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    progress = (values - start) / (target - start)
    low, high = (np.argmax(progress >= level) for level in (RISE_LOW, RISE_HIGH))
    outside = np.flatnonzero(np.abs(progress - 1) > SETTLE_BAND)
    if outside.size == 0:
        settling_time = 0.0
    elif outside[-1] == len(times) - 1:
        settling_time = float("inf")
    else:
        settling_time = times[outside[-1] + 1] - times[0]
    tail = values[-max(1, int(len(values) * TAIL_FRACTION)) :]
    return {
        "rise_time": float(times[high] - times[low]),
        "overshoot_percent": float(100 * max(0.0, progress.max() - 1)),
        "settling_time": float(settling_time),
        "steady_state_error": float(abs(tail.mean() - target)),
    }
