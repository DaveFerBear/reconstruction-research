"""Crowding: collapse the vertical gap between an adjacent pair of nodes."""

from __future__ import annotations

from typing import Any

from evals.common import MODE_DEFINITIONS, bbox, intersects, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


GAP_MIN = 20
GAP_MAX = 220


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    nodes = spec.get("nodes", [])
    if len(nodes) < 2:
        return None

    # Order by top edge, then look for a vertically-adjacent pair separated
    # by GAP_MIN..GAP_MAX with horizontal overlap (so the gap is *meaningful*
    # — not just whitespace beside a column).
    ordered = sorted(enumerate(nodes), key=lambda kv: (int(kv[1]["y"]), int(kv[1]["x"])))
    for k in range(len(ordered) - 1):
        ai, a = ordered[k]
        bi, b = ordered[k + 1]
        ax, ay, aw, ah = bbox(a)
        bx, by, bw, bh = bbox(b)
        gap = by - (ay + ah)
        if not (GAP_MIN <= gap <= GAP_MAX):
            continue
        # Require horizontal overlap so collapsing the gap is visually meaningful.
        if not (ax < bx + bw and ax + aw > bx):
            continue
        # Don't collapse pairs that already intersect (shouldn't happen, but defensive).
        if intersects(bbox(a), bbox(b)):
            continue
        new_y = ay + ah  # touching, no breathing room
        old_y = b["y"]
        spec["nodes"][bi] = {**b, "y": new_y}
        return Corruption(
            spec=spec,
            description=(
                f"node[{bi}] moved up: y {old_y}->{new_y} (collapsed {gap}px gap "
                f"below node[{ai}])"
            ),
            changed_node_indices=[bi],
        )

    return None


_DEF = MODE_DEFINITIONS["crowding"]
CORRUPTER = Corrupter(
    id="crowding",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
