"""The container registry: what is known about each type, and what to do with a stranger."""

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from beverage_disposal.core.config import from_mapping
from beverage_disposal.core.containers import ContainerRecord


@dataclass(frozen=True)
class EnrolmentRequest:
    """A container type nobody has described yet, waiting for the owner to enrol it."""

    type_id: str
    observed_mass: float  # kg, as weighed when it was met


@dataclass(frozen=True)
class Registry:
    """Known container types plus the unknown ones met so far. Never changed in place."""

    records: dict[str, ContainerRecord]
    pending: tuple[EnrolmentRequest, ...] = ()

    def lookup(self, type_id: str) -> ContainerRecord | None:
        return self.records.get(type_id)

    def meet(self, type_id: str, observed_mass: float) -> tuple[ContainerRecord | None, "Registry"]:
        """The record for `type_id`, or None and a registry that now holds a request to enrol it."""
        record = self.lookup(type_id)
        if record is not None:
            return record, self
        others = tuple(r for r in self.pending if r.type_id != type_id)
        request = EnrolmentRequest(type_id=type_id, observed_mass=observed_mass)
        return None, replace(self, pending=(*others, request))

    def enrol(self, record: ContainerRecord) -> "Registry":
        """A registry with `record` added and its request, if any, cleared."""
        if record.type_id in self.records:
            raise ValueError(f"{record.type_id} is already in the registry")
        pending = tuple(r for r in self.pending if r.type_id != record.type_id)
        return replace(self, records={**self.records, record.type_id: record}, pending=pending)


def load_registry(path: str | Path) -> Registry:
    """Read a registry file: one table per container type, the table name being the type id."""
    with open(path, "rb") as file:
        tables = tomllib.load(file)
    return Registry(
        records={
            type_id: from_mapping(ContainerRecord, {"type_id": type_id, **table}, type_id)
            for type_id, table in tables.items()
        }
    )
