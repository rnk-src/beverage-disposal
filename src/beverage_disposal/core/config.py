"""Scene settings as frozen dataclasses, read from a TOML file."""

import tomllib
from dataclasses import dataclass, fields, is_dataclass
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
class SceneConfig:
    container_x: float
    container_y: float
    fill_ml: float
    table: TableConfig
    container: ContainerSpec


def from_mapping(cls, data, where="scene"):
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


def load_scene_config(path):
    with open(path, "rb") as file:
        return from_mapping(SceneConfig, tomllib.load(file))
