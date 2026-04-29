"""Overlap: move one TextNode so its bbox intersects another TextNode's bbox."""

from __future__ import annotations

from typing import Any

from evals.common import MODE_DEFINITIONS, bbox, intersects, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    text_indices = [
        i for i, n in enumerate(spec.get("nodes", [])) if n.get("type") == "text"
    ]
    if len(text_indices) < 2:
        return None

    # Find the first ordered pair (a, b) whose bboxes don't currently overlap.
    for ai in text_indices:
        for bi in text_indices:
            if ai >= bi:
                continue
            a = spec["nodes"][ai]
            b = spec["nodes"][bi]
            if intersects(bbox(a), bbox(b)):
                continue
            # Move b so its center sits at a's center (clamped to canvas).
            ax, ay, aw, ah = bbox(a)
            _, _, bw, bh = bbox(b)
            new_x = max(0, ax + aw // 2 - bw // 2)
            new_y = max(0, ay + ah // 2 - bh // 2)
            old_x, old_y = b["x"], b["y"]
            spec["nodes"][bi] = {**b, "x": new_x, "y": new_y}
            return Corruption(
                spec=spec,
                description=(
                    f"node[{bi}] moved ({old_x},{old_y}) -> ({new_x},{new_y}) "
                    f"to overlap node[{ai}]"
                ),
                changed_node_indices=[bi],
            )

    return None  # every pair already overlaps — no clean way to induce


_DEF = MODE_DEFINITIONS["overlap"]
CORRUPTER = Corrupter(
    id="overlap",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
