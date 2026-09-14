"""Forward and inverse kinematics for the SO-101 arm's pitch joints
(shoulder_lift_joint, elbow_flex_joint, wrist_flex_joint).

This exists because the pick-and-lift descent used to move these three
joints by an ad hoc "equal and opposite fold" heuristic instead of real
IK, which only traced a straight vertical line over narrow, accidental
sub-ranges of that heuristic and produced a mostly-horizontal reach
elsewhere -- shoving the can out of the way before the gripper ever
closed. See commit-notes/10-iteration-3-pick-and-lift.md for the
investigation that found this.

Geometry (link lengths, base angles) below is derived from
so_arm101_macro.xacro's joint origins and verified against live TF
measurements from the real simulator (see that same commit-notes entry) --
these are not independently-guessed numbers.

Why this works as a classic 3-link planar arm: shoulder_lift, elbow_flex
and wrist_flex all rotate about the same world-frame axis once
shoulder_pan is fixed (confirmed numerically, not assumed), so together
they move the gripper within a single vertical plane. shoulder_pan (base
rotation, chooses which vertical plane) is left to existing code -- this
module only solves the 2-D problem *within* that plane: given a target
(radius, height) and a target gripper pitch, find the three joint angles.
"""
import math

# Joint origins from so_arm101_description/urdf/so_arm101_macro.xacro,
# each as (xyz translation, rpy) of the joint frame relative to its
# parent link, exactly as URDF defines it. shoulder_pan's *rotation* is
# handled by the caller (it just picks which vertical plane we work in),
# but its fixed origin transform still has to be included at theta=0 --
# that origin's rpy is close to a 180-degree flip, a real part of how
# shoulder_link is mounted on base_link, not something the caller
# substitutes for.
_J_SHOULDER_PAN = ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159))
_J_SHOULDER_LIFT = ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0))
_J_ELBOW_FLEX = ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708))
_J_WRIST_FLEX = ((-0.1349, 0.0052, 3.62355e-17), (-1.41553e-15, 4.91611e-15, -1.57))
_J_WRIST_ROLL = ((0.0, -0.0611, 0.0181), (-1.57, 3.14, 0.0))
_J_GRIPPER_FRAME = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))

# Joint limits (from the same xacro), used to reject unreachable targets
# with a clear error instead of silently commanding an out-of-range angle.
SHOULDER_LIFT_LIMITS = (-1.74533, 1.74533)
ELBOW_FLEX_LIMITS = (-1.69, 1.54)
WRIST_FLEX_LIMITS = (-1.6, 1.6)


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def _matmul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0])))
        for i in range(len(a)))


def _rpy_matrix(roll, pitch, yaw):
    # URDF's <origin rpy="r p y"/> is fixed-axis roll-pitch-yaw:
    # R = Rz(yaw) * Ry(pitch) * Rx(roll).
    return _matmul(_rot_z(yaw), _matmul(_rot_y(pitch), _rot_x(roll)))


def _homogeneous(rot, xyz):
    return (rot[0] + (xyz[0],), rot[1] + (xyz[1],), rot[2] + (xyz[2],), (0, 0, 0, 1))


def _joint_transform(origin_xyz, origin_rpy, theta):
    """Transform from a joint's parent frame to its child frame: translate
    and rotate to the joint's own origin, then rotate by theta about the
    joint's local Z axis (every joint on this arm uses axis="0 0 1")."""
    origin = _homogeneous(_rpy_matrix(*origin_rpy), origin_xyz)
    spin = _homogeneous(_rot_z(theta), (0, 0, 0))
    return _matmul(origin, spin)


def _translation_of(transform):
    return (transform[0][3], transform[1][3], transform[2][3])


def _fk_chain(shoulder_lift, elbow_flex, wrist_flex, wrist_roll):
    """Returns the world-frame (x, y, z) of the shoulder_lift pivot, elbow
    pivot, wrist pivot, and gripper_frame_link, all with shoulder_pan
    fixed at 0 (the caller handles the base rotation separately)."""
    t = _joint_transform(*_J_SHOULDER_PAN, 0.0)

    t_before_lift = _matmul(t, _homogeneous(_rpy_matrix(*_J_SHOULDER_LIFT[1]), _J_SHOULDER_LIFT[0]))
    p_shoulder = _translation_of(t_before_lift)

    t = _matmul(t, _joint_transform(*_J_SHOULDER_LIFT, shoulder_lift))
    t_before_elbow = _matmul(t, _homogeneous(_rpy_matrix(*_J_ELBOW_FLEX[1]), _J_ELBOW_FLEX[0]))
    p_elbow = _translation_of(t_before_elbow)

    t = _matmul(t, _joint_transform(*_J_ELBOW_FLEX, elbow_flex))
    t_before_wrist = _matmul(t, _homogeneous(_rpy_matrix(*_J_WRIST_FLEX[1]), _J_WRIST_FLEX[0]))
    p_wrist = _translation_of(t_before_wrist)

    t = _matmul(t, _joint_transform(*_J_WRIST_FLEX, wrist_flex))
    t = _matmul(t, _joint_transform(*_J_WRIST_ROLL, wrist_roll))
    t = _matmul(t, _joint_transform(*_J_GRIPPER_FRAME, 0.0))
    p_gripper = _translation_of(t)

    return p_shoulder, p_elbow, p_wrist, p_gripper


def _plane_angle(p_from, p_to):
    """Angle, in the (radius, height) = (x, z) plane, of the vector from
    p_from to p_to."""
    return math.atan2(p_to[2] - p_from[2], p_to[0] - p_from[0])


def _planar_dist(p_from, p_to):
    """Distance between two points *projected onto the (x, z) plane* --
    not full 3-D distance. shoulder/elbow/wrist share an identical Y
    (confirmed against live TF), so this equals the 3-D distance for
    links 1 and 2, but the wrist->gripper offset has a real ~1cm
    out-of-plane Y component (a fixed sideways offset, not something that
    varies with any joint value in this model -- see forward_kinematics'
    docstring), so link 3 must use the planar projection: the (x, z)
    angle math elsewhere in this module only ever rotates that
    projection, not the true 3-D vector, and mixing the two in the same
    formula was found (via a failing FK/IK round-trip test) to introduce
    a small but real ~0.3mm systematic error."""
    return math.hypot(p_to[0] - p_from[0], p_to[2] - p_from[2])


def _link_geometry(wrist_roll):
    """Derives the 3-link planar-arm constants (link lengths and each
    link's angle at joint value 0) directly from forward kinematics,
    rather than hardcoding separately-computed numbers that could drift
    out of sync with the URDF geometry above."""
    p_shoulder, p_elbow, p_wrist, p_gripper = _fk_chain(0.0, 0.0, 0.0, wrist_roll)
    l1 = _planar_dist(p_shoulder, p_elbow)
    l2 = _planar_dist(p_elbow, p_wrist)
    l3 = _planar_dist(p_wrist, p_gripper)
    seg1_0 = _plane_angle(p_shoulder, p_elbow)
    seg2_0 = _plane_angle(p_elbow, p_wrist)
    seg3_0 = _plane_angle(p_wrist, p_gripper)
    return p_shoulder, l1, l2, l3, seg1_0, seg2_0, seg3_0


def forward_kinematics(shoulder_lift, elbow_flex, wrist_flex, wrist_roll):
    """Given the three pitch-joint values (plus the fixed wrist_roll used
    throughout a grasp), returns (radius, height, pitch): the gripper's
    position in the arm's vertical plane (radius = distance from the
    shoulder_pan axis, height = height above base_link) and its pitch (the
    angle, in that same plane, of the wrist->gripper direction).

    Note: this ignores the arm's small (~1cm) constant sideways offset
    from a perfectly flat plane (see commit-notes/10's "lateral offset"
    finding) -- same simplification the old fold-based heuristic made, so
    this is a strict improvement on it, not a regression.
    """
    p_shoulder, p_elbow, p_wrist, p_gripper = _fk_chain(shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
    radius = p_gripper[0]
    height = p_gripper[2]
    pitch = _plane_angle(p_wrist, p_gripper)
    return radius, height, pitch


def solve_arm_ik(target_radius, target_height, target_pitch, wrist_roll, elbow_up=True,
                  clamp_near_limits=0.0):
    """Closed-form inverse kinematics for shoulder_lift/elbow_flex/wrist_flex.

    Solves for the three joint angles that place the gripper at
    (target_radius, target_height) in the arm's vertical plane, pointing
    at target_pitch, given a fixed wrist_roll (this project always holds
    wrist_roll constant during a grasp approach).

    Raises ValueError if the target is out of reach (too far, too close,
    or requires an angle outside a joint's mechanical limits).

    clamp_near_limits: if a solved joint value is outside its limit by no
    more than this many radians, clamp it back to the limit instead of
    raising. Off (0.0) by default. Real use case: a straight Cartesian
    line between two endpoints that each sit exactly at a joint's limit
    (e.g. this arm's own GRASP and HOVER poses both use wrist_flex=1.6,
    its hard maximum) can bulge a fraction of a degree past that limit at
    a point in between, since the limit itself is a curve in this plane
    and the interpolated path is a straight chord across it -- a real
    geometric fact, not a bug, and not worth rejecting a whole path over.
    """
    p_shoulder, l1, l2, l3, seg1_0, seg2_0, seg3_0 = _link_geometry(wrist_roll)

    # Step 1: knowing the desired gripper pitch, subtract link 3 to get the
    # position the wrist itself (not the gripper tip) needs to reach --
    # the classic "wrist center" reduction that turns a 3-link
    # position+orientation problem into a 2-link position-only problem.
    wrist_x = target_radius - l3 * math.cos(target_pitch)
    wrist_z = target_height - l3 * math.sin(target_pitch)

    dx = wrist_x - p_shoulder[0]
    dz = wrist_z - p_shoulder[2]
    r_sq = dx * dx + dz * dz
    r = math.sqrt(r_sq)
    # Tiny floating-point tolerance, same idea (and same 1e-6 size) as
    # limit_tol below: a target reconstructed from this arm's own true
    # max/min reach (e.g. a path endpoint computed via this module's own
    # forward_kinematics) can land a hair outside [abs(l1-l2), l1+l2]
    # purely from floating-point roundoff in whatever arithmetic produced
    # it, not because it's actually unreachable. Without this, a
    # mathematically-exact boundary target could spuriously raise.
    reach_tol = 1e-6
    if r > l1 + l2 + reach_tol or r < abs(l1 - l2) - reach_tol:
        raise ValueError(
            f'target (radius={target_radius:.4f}, height={target_height:.4f}, '
            f'pitch={target_pitch:.4f}) is out of reach: wrist-center distance '
            f'{r:.4f}m is outside [{abs(l1 - l2):.4f}, {l1 + l2:.4f}]m')

    # Law of cosines for the angle between link 1 and link 2 (elbow
    # bend). Two solutions (elbow up/down) -- elbow_up picks the sign.
    cos_gamma = (r_sq - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    cos_gamma = max(-1.0, min(1.0, cos_gamma))
    gamma = math.acos(cos_gamma) if elbow_up else -math.acos(cos_gamma)

    # Standard 2-link planar IK: absolute angle of link 1, then link 2 is
    # link 1's angle plus the elbow bend.
    a1 = math.atan2(dz, dx) - math.atan2(l2 * math.sin(gamma), l1 + l2 * math.cos(gamma))
    a2 = a1 + gamma
    a3 = target_pitch

    # These joints' angle-to-plane-direction relationship is inverted
    # (verified against live TF: rotating shoulder_lift by +theta rotates
    # the plane angle by -theta -- see commit-notes/10), and each later
    # joint's plane angle is measured relative to the cumulative angle of
    # every joint before it, so unwinding back to raw joint values means
    # subtracting off what the earlier joints already contributed.
    solved = {
        'shoulder_lift': (seg1_0 - a1, SHOULDER_LIFT_LIMITS),
        'elbow_flex': None,  # filled in below, depends on shoulder_lift
        'wrist_flex': None,  # filled in below, depends on both
    }
    solved['elbow_flex'] = (seg2_0 - solved['shoulder_lift'][0] - a2, ELBOW_FLEX_LIMITS)
    solved['wrist_flex'] = (
        seg3_0 - solved['shoulder_lift'][0] - solved['elbow_flex'][0] - a3, WRIST_FLEX_LIMITS)

    # Small tolerance beyond clamp_near_limits: a target that is exactly
    # at a joint's hard limit can still land a hair outside it after the
    # chain of trig operations above, purely from floating-point roundoff.
    limit_tol = 1e-6
    result = {}
    for name, (value, limits) in solved.items():
        if value < limits[0]:
            if limits[0] - value <= clamp_near_limits + limit_tol:
                value = limits[0]
            else:
                raise ValueError(
                    f'{name}={value:.4f} rad is outside its joint limit {limits} '
                    f'for target (radius={target_radius:.4f}, height={target_height:.4f}, '
                    f'pitch={target_pitch:.4f})')
        elif value > limits[1]:
            if value - limits[1] <= clamp_near_limits + limit_tol:
                value = limits[1]
            else:
                raise ValueError(
                    f'{name}={value:.4f} rad is outside its joint limit {limits} '
                    f'for target (radius={target_radius:.4f}, height={target_height:.4f}, '
                    f'pitch={target_pitch:.4f})')
        result[name] = value

    return result['shoulder_lift'], result['elbow_flex'], result['wrist_flex']
