"""Corrupter contract — applied to real specs from `datasets/specs/`.

Each corrupter takes a parsed spec dict (the JSON form, not the Pydantic Spec)
and either returns a `Corruption` describing one isolated mutation that
induces the target failure mode, or `None` if the spec lacks the structure to
corrupt cleanly. None means "skip this (spec, mode) pair" — not an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Corruption:
    spec: dict[str, Any]              # mutated spec dict (deep-copied from input)
    description: str                  # one-line human-readable change
    changed_node_indices: list[int]   # which spec.nodes entries were modified


@dataclass(frozen=True)
class Corrupter:
    id: str                           # matches a FailureMode.id
    name: str
    description: str                  # judge-prompt definition
    apply: Callable[[dict[str, Any]], Corruption | None]
