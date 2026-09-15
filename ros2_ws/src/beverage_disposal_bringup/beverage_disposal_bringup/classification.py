# Classifies fullness from the can's real simulated mass, not joint effort.
#
# The original plan for this was to read gripper/arm joint effort while
# holding the can and infer weight from that ("ad hoc weighing"). That
# premise turned out to be physically false in this pipeline, for two
# reasons found while building Iteration 3 (pick + lift):
#
#   - gripper_action_server.py's contact-detection controller outputs a
#     *fixed* commanded effort (contact_hold_effort) once real contact is
#     detected, regardless of what's actually gripped -- it's a control
#     target, not a sensed reaction force.
#   - pick_and_lift.py's kinematic-lock mechanism (needed to make the lift
#     itself reliable -- see kinematic_grasp.py) directly overrides the
#     can's world pose every tick, so its real mass never loads back
#     through the gripper or arm joints during a lift at all.
#
# So no joint-effort signal in this pipeline actually varies with the held
# object's mass. Rather than fake a sensor reading that isn't physically
# meaningful, this reads the simulated object's real, known mass directly
# -- an honest, documented ground-truth simplification, the same category
# this project already used for the pose-based ultrasonic sensor and for
# kinematic snap-grasp itself. See pick_and_lift.py's can_mass_kg parameter
# for how that ground-truth value actually reaches this function.
EMPTY_CAN_MASS_KG = 0.014  # a real empty 12oz aluminum can
FULL_CAN_MASS_KG = 0.37  # ~355mL liquid + the empty shell, rounded

# Any value strictly between the two constants above is equally correct --
# this is a noiseless ground-truth read, not a real sensor with measurement
# error, so there's no calibration sweet spot to find. The midpoint just
# leaves symmetric headroom if a third mass variant is ever added later.
FULLNESS_MASS_THRESHOLD_KG = 0.19


def classify_fullness(mass_kg):
    if mass_kg >= FULLNESS_MASS_THRESHOLD_KG:
        return 'full'
    return 'empty'
