"""Overflow: pick a TextNode and shrink its container so the text spills out."""

from __future__ import annotations

from typing import Any

from evals.common import MODE_DEFINITIONS, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    # Pick the first TextNode with non-trivial content. We deterministically
    # take the *first* text-heavy node so the corpus is reproducible.
    candidates: list[tuple[int, dict]] = []
    for i, node in enumerate(spec.get("nodes", [])):
        if node.get("type") != "text":
            continue
        text = node.get("text") or ""
        if len(text.strip()) < 4:
            continue
        if int(node.get("width", 0)) < 60 or int(node.get("font_size", 0)) < 12:
            continue
        candidates.append((i, node))
    if not candidates:
        return None

    # Prefer nodes whose text fits comfortably in a wide container (so cutting
    # the width in half produces visible overflow).
    def _slack(item: tuple[int, dict]) -> float:
        _, n = item
        approx_required = len(n["text"]) * float(n["font_size"]) * 0.55
        return float(n["width"]) - approx_required

    candidates.sort(key=lambda x: (-_slack(x), x[0]))
    idx, node = candidates[0]
    old_width = int(node["width"])
    new_width = max(40, old_width // 3)
    spec["nodes"][idx] = {**node, "width": new_width}

    return Corruption(
        spec=spec,
        description=f"node[{idx}].width: {old_width} -> {new_width} (text now overflows)",
        changed_node_indices=[idx],
    )


_DEF = MODE_DEFINITIONS["overflow"]
CORRUPTER = Corrupter(
    id="overflow",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
