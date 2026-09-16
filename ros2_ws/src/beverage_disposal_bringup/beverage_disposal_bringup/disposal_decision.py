# Decides what to do with a can given its classified fullness (Iteration 4):
# an empty can goes in the bin, a full one is set aside (lowered back down
# near its original pickup pose, per the user's explicit choice -- see
# CLAUDE.md's Iteration 5 entry) rather than processed further.
#
# decide_disposal only ever runs on an already-validated fullness value in
# the real pipeline: pick_and_lift.py computes fullness = classify_fullness(
# self.can_mass_kg) if success else None, then only calls decide_disposal
# inside that same `if success:` branch -- so fullness is always exactly
# 'empty' or 'full' by the time it gets here. Raising ValueError on anything
# else is a contract check (a violated precondition, same as
# kinematics.solve_arm_ik raising on an out-of-reach target), not a real
# production path -- callers don't need a try/except around this call.


def decide_disposal(fullness):
    if fullness == 'empty':
        return 'bin'
    if fullness == 'full':
        return 'set_aside'
    raise ValueError(f'unrecognized fullness value: {fullness!r}')
