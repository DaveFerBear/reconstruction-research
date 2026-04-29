"""Nonsensical hierarchy: shrink the title to roughly the size of a body element.

Pared back from a full size-swap. The title is reduced so it's no longer
visually dominant, but body text is left untouched — softer hierarchy
inversion that still reads as a real failure mode."""

from __future__ import annotations

from typing import Any

from evals.common import MODE_DEFINITIONS, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


SIZE_RATIO = 1.6  # title must be at least 1.6x the body font size to qualify


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    text_nodes = [
        (i, n) for i, n in enumerate(spec.get("nodes", [])) if n.get("type") == "text"
    ]
    if len(text_nodes) < 2:
        return None

    text_nodes.sort(key=lambda t: -int(t[1].get("font_size", 0)))
    title_idx, title = text_nodes[0]
    title_size = int(title.get("font_size", 0))

    # Find a body small enough that promoting the title down to its size
    # produces a visible hierarchy inversion.
    body_size: int | None = None
    for _, n in text_nodes[1:]:
        candidate = int(n.get("font_size", 0))
        if candidate > 0 and title_size / candidate >= SIZE_RATIO:
            body_size = candidate
            break
    if body_size is None:
        return None

    # Pare back: drop the title to body-size. Don't grow the body.
    spec["nodes"][title_idx] = {**title, "font_size": body_size}
    return Corruption(
        spec=spec,
        description=(
            f"node[{title_idx}].font_size: {title_size} -> {body_size} "
            f"(title shrunk to body size)"
        ),
        changed_node_indices=[title_idx],
    )


_DEF = MODE_DEFINITIONS["nonsensical_hierarchy"]
CORRUPTER = Corrupter(
    id="nonsensical_hierarchy",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
