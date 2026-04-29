"""Failure-mode contract.

Each module under `evals.failure_modes` exposes a `MODE: FailureMode` constant.
A `FailureMode` produces a list of `Variant`s, where each variant is a
(bad, good) pair of specs that differ only in the failure invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Variant:
    bad_spec: dict[str, Any]
    bad_svgs: dict[str, str]
    good_spec: dict[str, Any]
    good_svgs: dict[str, str]


@dataclass(frozen=True)
class FailureMode:
    id: str
    name: str
    description: str  # one-sentence definition shown to the VLM judge
    generate: Callable[[], list[Variant]]
