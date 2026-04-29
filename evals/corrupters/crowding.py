"""Crowding: collapse the vertical gaps in a stack of adjacent nodes.

Walks down a chain of vertically adjacent, horizontally-overlapping nodes,
collapsing each gap to zero. A 3+ node chain produces a visibly crammed
section; a 2-node fallback collapses the single pair.
"""

from __future__ import annotations

from typing import Any

from evals.common import MODE_DEFINITIONS, bbox, intersects, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


GAP_MIN = 12
GAP_MAX = 260


def _h_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, _, aw, _ = a
    bx, _, bw, _ = b
    return ax < bx + bw and ax + aw > bx


def _build_chain(nodes: list[dict], start: int) -> list[int]:
    """Return indices of a vertical chain starting at `start`. Each subsequent
    node sits below the previous, horizontally overlaps it, and is GAP_MIN..MAX
    pixels away."""
    chain = [start]
    cur = nodes[start]
    while True:
        cur_box = bbox(cur)
        cur_bottom = cur_box[1] + cur_box[3]
        # Find the nearest node strictly below the current one with horizontal overlap
        # and a gap inside the threshold band.
        best: int | None = None
        best_gap = None
        for i, n in enumerate(nodes):
            if i in chain:
                continue
            nb = bbox(n)
            gap = nb[1] - cur_bottom
            if not (GAP_MIN <= gap <= GAP_MAX):
                continue
            if not _h_overlap(cur_box, nb):
                continue
            if intersects(cur_box, nb):
                continue
            if best is None or gap < best_gap:
                best, best_gap = i, gap
        if best is None:
            break
        chain.append(best)
        cur = nodes[best]
    return chain


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    nodes = spec.get("nodes", [])
    if len(nodes) < 2:
        return None

    # Try every possible starting node; keep the longest chain we find.
    best_chain: list[int] = []
    for i in range(len(nodes)):
        chain = _build_chain(nodes, i)
        if len(chain) > len(best_chain):
            best_chain = chain

    if len(best_chain) < 2:
        return None

    # Collapse every gap in the chain to zero.
    changed: list[int] = []
    for k in range(len(best_chain) - 1):
        upper_idx = best_chain[k]
        lower_idx = best_chain[k + 1]
        upper = spec["nodes"][upper_idx]
        lower = spec["nodes"][lower_idx]
        ux, uy, uw, uh = bbox(upper)
        new_y = uy + uh
        if int(lower["y"]) != new_y:
            spec["nodes"][lower_idx] = {**lower, "y": new_y}
            changed.append(lower_idx)

    if not changed:
        return None

    return Corruption(
        spec=spec,
        description=(
            f"collapsed {len(changed)} vertical gap(s) in a chain of "
            f"{len(best_chain)} nodes ({' -> '.join(str(i) for i in best_chain)})"
        ),
        changed_node_indices=changed,
    )


_DEF = MODE_DEFINITIONS["crowding"]
CORRUPTER = Corrupter(
    id="crowding",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
