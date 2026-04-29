"""Misalignment: find a node centered inside another and shift it off-center."""

from __future__ import annotations

from typing import Any

from evals.common import MODE_DEFINITIONS, bbox, center, contains, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


CENTER_TOLERANCE = 0.10  # child center must be within 10% of parent center


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    nodes = spec.get("nodes", [])
    if len(nodes) < 2:
        return None

    for ci, child in enumerate(nodes):
        cb = bbox(child)
        for pi, parent in enumerate(nodes):
            if pi == ci:
                continue
            pb = bbox(parent)
            if not contains(pb, cb):
                continue
            # Parent must be meaningfully bigger than child or there's no
            # room for misalignment to read.
            if pb[2] < cb[2] * 1.4 or pb[3] < cb[3] * 1.4:
                continue
            cx_p, cy_p = center(pb)
            cx_c, cy_c = center(cb)
            if (
                abs(cx_c - cx_p) > pb[2] * CENTER_TOLERANCE
                or abs(cy_c - cy_p) > pb[3] * CENTER_TOLERANCE
            ):
                continue
            # Centered child found — shift it toward the upper-right corner.
            dx = int(pb[2] * 0.25)
            dy = -int(pb[3] * 0.20)
            old_x, old_y = child["x"], child["y"]
            new_x = old_x + dx
            new_y = old_y + dy
            spec["nodes"][ci] = {**child, "x": new_x, "y": new_y}
            return Corruption(
                spec=spec,
                description=(
                    f"node[{ci}] shifted ({old_x},{old_y}) -> ({new_x},{new_y}) "
                    f"inside parent node[{pi}]"
                ),
                changed_node_indices=[ci],
            )

    return None


_DEF = MODE_DEFINITIONS["misalignment"]
CORRUPTER = Corrupter(
    id="misalignment",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
