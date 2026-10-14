"""Numerical inverse kinematics for the SO-101 tip, in plain NumPy."""

from dataclasses import dataclass

import numpy as np

from beverage_disposal.core.kinematics import HOME, fk, jacobian, load_chain, pitch

# Pan, lift, elbow and wrist_flex move the tip; the wrist roll stays where the caller left it.
IK_JOINTS = 4
DAMPING = 0.01
MAX_STEP = 0.2
POSITION_TOLERANCE = 1e-4
PITCH_TOLERANCE = 1e-3
MAX_ITERATIONS = 100
# Singular values below this count as zero when finding the motions that leave the tip in place.
RANK_TOLERANCE = 1e-6
# An iteration that moves no joint further than this has nothing left to give: a stationary point,
# such as the best pitch the arm can manage when the asked-for one is out of reach.
STALL_STEP = 1e-8
_LOW, _HIGH = np.array([joint["range"] for joint in load_chain()["joints"]]).T[:, :IK_JOINTS]


@dataclass(frozen=True)
class IKResult:
    angles: np.ndarray
    position_error: float
    pitch_error: float
    converged: bool
    iterations: int


def _turn(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _damped_pinv(matrix, damping):
    return matrix.T @ np.linalg.inv(matrix @ matrix.T + damping**2 * np.eye(matrix.shape[0]))


def _null_space_basis(matrix):
    """Rows spanning the joint motions that leave the task matrix's output unchanged."""
    _, singular_values, rows = np.linalg.svd(matrix)
    rank = int(np.sum(singular_values > RANK_TOLERANCE))
    return rows[rank:]


def _limit_norm(vector, limit):
    return vector * min(1.0, limit / max(np.linalg.norm(vector), 1e-300))


def _fit_secondary(primary, secondary, limit):
    """Shrink `secondary` until primary + secondary fit in `limit`; `primary` is never touched."""
    a = secondary @ secondary
    if a == 0.0:
        return secondary
    b = primary @ secondary
    c = primary @ primary - limit**2
    if c >= 0.0:
        return np.zeros_like(secondary)
    return secondary * min(1.0, (-b + np.sqrt(b * b - a * c)) / a)


def _step(jac, position_error, pitch_error, active, max_step, damping):
    """One damped, capped step over the `active` joints: position first, pitch in what is left."""
    jac = jac[:, active]
    position_pinv = _damped_pinv(jac[:3], damping)
    step = _limit_norm(position_pinv @ position_error, max_step)
    if pitch_error is not None:
        # The exact null space of the position Jacobian, not 1 - J+ J of the damped inverse: the
        # damped version leaks into the position task and the leak piles up.
        free = _null_space_basis(jac[:3])
        remaining = np.array([pitch_error - jac[3] @ step])
        secondary = free.T @ _damped_pinv(jac[3:] @ free.T, damping) @ remaining
        step = step + _fit_secondary(step, secondary, max_step)
    full = np.zeros(IK_JOINTS)
    full[active] = step
    return full


def solve_ik(
    target,
    target_pitch=np.pi / 2,
    start=HOME,
    *,
    max_iterations=MAX_ITERATIONS,
    max_step=MAX_STEP,
    damping=DAMPING,
    position_tolerance=POSITION_TOLERANCE,
    pitch_tolerance=PITCH_TOLERANCE,
):
    """Joint angles that put the TCP at `target` (base frame, metres) with the given pitch.

    Position has strict priority: the pitch correction only uses the freedom the position task
    leaves (its null space), and when the step has to be capped it is the pitch part that shrinks.
    A joint that a step would push past its limit is parked on the limit and the step is solved
    again over the other joints; clipping alone would let the pitch task drag the tip off target.
    `target_pitch=None` drops the pitch task. Never raises for a target out of reach: it returns
    the best effort with `converged=False`.
    """
    angles = np.array(start, dtype=float)

    def errors():
        position = target - fk(angles)[:3, 3]
        turned = None if target_pitch is None else _turn(target_pitch - pitch(angles))
        return position, turned

    def advance(with_pitch):
        position_error, pitch_error = errors()
        active = np.ones(IK_JOINTS, dtype=bool)
        while True:
            step = _step(
                jacobian(angles)[:, :IK_JOINTS],
                position_error,
                pitch_error if with_pitch else None,
                active,
                max_step,
                damping,
            )
            proposed = angles[:IK_JOINTS] + step
            outside = active & ((proposed < _LOW) | (proposed > _HIGH))
            if not outside.any():
                angles[:IK_JOINTS] = proposed
                return
            angles[:IK_JOINTS][outside] = np.clip(proposed, _LOW, _HIGH)[outside]
            active &= ~outside
            if not active.any():
                return
            position_error, pitch_error = errors()

    iterations = 0
    while iterations < max_iterations:
        position_error, pitch_error = errors()
        pitch_ok = pitch_error is None or abs(pitch_error) < pitch_tolerance
        if np.linalg.norm(position_error) < position_tolerance and pitch_ok:
            break
        iterations += 1
        before = angles.copy()
        advance(with_pitch=target_pitch is not None)
        if target_pitch is not None:
            # A pitch step only keeps the tip in place to first order; a position-only step after it
            # takes out the second-order drift, so every iteration ends with the position on target.
            advance(with_pitch=False)
        if np.max(np.abs(angles - before)) < STALL_STEP:
            break
    position_error, pitch_error = errors()
    position_error = float(np.linalg.norm(position_error))
    pitch_error = 0.0 if pitch_error is None else float(abs(pitch_error))
    converged = position_error < position_tolerance and pitch_error < pitch_tolerance
    return IKResult(angles, position_error, pitch_error, converged, iterations)
