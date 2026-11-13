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
class SceneConfig:
    container_x: float
    container_y: float
    fill_ml: float
    table: TableConfig
    container: ContainerSpec
    bin: BinConfig
    set_aside: SetAsideConfig
    pad: PadConfig
    spawn: SpawnConfig
    reach: ReachConfig


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
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected a table")
        return from_mapping(hint, value, where)
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, hint) or isinstance(value, bool):
        raise ValueError(f"{where}: expected {hint.__name__}, got {value!r}")
    return value


def load_scene_config(path: str | Path) -> SceneConfig:
    with open(path, "rb") as file:
        return from_mapping(SceneConfig, tomllib.load(file))
