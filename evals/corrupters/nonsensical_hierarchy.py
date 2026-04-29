"""Nonsensical hierarchy: swap font_size between the largest and a smaller TextNode."""

from __future__ import annotations

from typing import Any

from evals.common import MODE_DEFINITIONS, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


SIZE_RATIO = 1.6  # title must be at least 1.6x the body font size


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    text_nodes = [
        (i, n) for i, n in enumerate(spec.get("nodes", [])) if n.get("type") == "text"
    ]
    if len(text_nodes) < 2:
        return None

    # Largest text by font_size (the de-facto "title")
    text_nodes.sort(key=lambda t: -int(t[1].get("font_size", 0)))
    title_idx, title = text_nodes[0]
    title_size = int(title.get("font_size", 0))

    # Find a body text small enough that swapping is conspicuous
    body_idx: int | None = None
    body: dict | None = None
    for i, n in text_nodes[1:]:
        body_size = int(n.get("font_size", 0))
        if body_size > 0 and title_size / body_size >= SIZE_RATIO:
            body_idx, body = i, n
            break
    if body_idx is None or body is None:
        return None

    body_size = int(body.get("font_size", 0))
    spec["nodes"][title_idx] = {**title, "font_size": body_size}
    spec["nodes"][body_idx] = {**body, "font_size": title_size}
    return Corruption(
        spec=spec,
        description=(
            f"swapped font_size: node[{title_idx}] {title_size}->{body_size}, "
            f"node[{body_idx}] {body_size}->{title_size}"
        ),
        changed_node_indices=sorted([title_idx, body_idx]),
    )


_DEF = MODE_DEFINITIONS["nonsensical_hierarchy"]
CORRUPTER = Corrupter(
    id="nonsensical_hierarchy",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
