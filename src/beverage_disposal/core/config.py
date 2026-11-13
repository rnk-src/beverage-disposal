"""Scene settings as frozen dataclasses, read from a TOML file."""

import tomllib
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

from beverage_disposal.core.containers import ContainerSpec


@dataclass(frozen=True)
class TableConfig:
    center_x: float
    half_x: float
    half_y: float
    thickness: float
    friction: float


@dataclass(frozen=True)
class BinConfig:
    """An open-top box; `inner_*` are the free floor size, all lengths in metres."""

    center_x: float
    center_y: float
    inner_x: float
    inner_y: float
    height: float  # the walls' top above the table
    wall: float
    floor: float
    friction: float


@dataclass(frozen=True)
class SetAsideConfig:
    """A marked flat region of the table, half sizes in metres."""

    center_x: float
    center_y: float
    half_x: float
    half_y: float


@dataclass(frozen=True)
class PadConfig:
    """The weighing pad's plate (geometry only), half sizes and height in metres."""

    center_x: float
    center_y: float
    half_x: float
    half_y: float
    height: float
    friction: float


@dataclass(frozen=True)
class SpawnConfig:
    """Where a container may start: a square of half-width `half_size` metres."""

    center_x: float
    center_y: float
    half_size: float


@dataclass(frozen=True)
class ReachConfig:
    """What the reach check asks of the arm."""

    min_pitch_deg: float  # the steepest-tilt limit: poses must be reachable at this pitch or more
    approach_clearance: float  # metres above a grasp or place pose for the approach
    lift_height: float  # metres a picked container is lifted
    release_clearance: float  # metres between the container's bottom and the bin wall's top


@dataclass(frozen=True)
class GraspConfig:
    """How the prescribed grasp moves; angles in radians, lengths in metres."""

    wrist_roll: float  # locked for the whole grasp; pi/2 makes the jaws close sideways
    pitch_deg: float  # tool tilt below horizontal, 90 is straight down
    open_angle: float  # gripper angle while the pads straddle the container
    close_target: float  # gripper target while squeezing (past the container, so it stalls on it)
    close_rate: float  # target decrease per 20 Hz control step
    clearance: float  # gap between the fixed pad and the container before closing
    force_limit: float  # gripper servo torque limit, N m (the vendored limit is 3.35)


@dataclass(frozen=True)
class GripperPadsConfig:
    """Two flat pads replace the jaw meshes' collision; see sim/pads.py."""

    design_angle: float  # gripper angle at which the two pad faces are parallel, rad
    thickness: float
    half_width: float
    half_height: float
    depth: float  # pad centre along the gripper's z axis (negative: towards the fingertips)
    friction: float
    torsional_friction: float  # resists the container twisting on the pads, metres (MuJoCo units)


@dataclass(frozen=True)
class PhysicsConfig:
    timestep: float  # seconds
    contact_time_constant: float  # MuJoCo solref[0], seconds: smaller is stiffer
    contact_damping_ratio: float  # MuJoCo solref[1]


@dataclass(frozen=True)
class SceneConfig:
    container_x: float
    container_y: float
    fill_ml: float
    container_type: str  # the registry table the container comes from
    table: TableConfig
    container: ContainerSpec
    bin: BinConfig
    set_aside: SetAsideConfig
    pad: PadConfig
    spawn: SpawnConfig
    reach: ReachConfig
    grasp: GraspConfig
    gripper_pads: GripperPadsConfig
    physics: PhysicsConfig


def from_mapping(cls: type, data: dict, where: str = "scene"):
    """Build dataclass `cls` from a mapping; unknown keys, missing keys and wrong types raise."""
    hints = get_type_hints(cls)
    unknown = sorted(set(data) - set(hints))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}")
    missing = sorted(f.name for f in fields(cls) if f.name not in data)
    if missing:
        raise ValueError(f"{where}: missing key(s) {missing}")
    return cls(**{name: _convert(hints[name], data[name], f"{where}.{name}") for name in data})


def _convert(hint, value, where):
    if is_dataclass(hint):
        if isinstance(value, hint):
            return value
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected a table")
        return from_mapping(hint, value, where)
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, hint) or (isinstance(value, bool) and hint is not bool):
        raise ValueError(f"{where}: expected {hint.__name__}, got {value!r}")
    return value


def load_scene_config(path: str | Path, registry_path: str | Path | None = None) -> SceneConfig:
    """Read a scene file; its container comes from the registry next to it unless one is given."""
    # Imported here because the registry module itself uses this module's loader.
    from beverage_disposal.core.registry import load_registry

    path = Path(path)
    with open(path, "rb") as file:
        data = tomllib.load(file)
    registry = load_registry(registry_path or path.parent / "containers.toml")
    type_id = data.get("container_type")
    record = registry.lookup(type_id) if isinstance(type_id, str) else None
    if record is None:
        raise ValueError(f"scene: container_type {type_id!r} is not in the registry")
    return from_mapping(SceneConfig, {**data, "container": record.spec})
