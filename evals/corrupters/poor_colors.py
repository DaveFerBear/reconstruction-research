"""Poor colors: replace every TextNode's color with a value from a clashing,
garish, uncoordinated palette.

Distinct from `inconsistency` (which mutates ONE item out of a uniform group)
and from `poor_contrast` (which makes ONE text label hard to read against the
bg). Here the entire color scheme of the design's text is intentionally bad —
every element gets a different high-saturation color from a palette designed
to vibrate and not harmonize.
"""

from __future__ import annotations

from typing import Any

from evals.common import MODE_DEFINITIONS, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


# A deliberately uncoordinated palette: full saturation, no shared hue or
# tonal grouping. Cycled across TextNodes so every adjacent pair of labels
# gets a clashing combination.
CLASHING_PALETTE: tuple[str, ...] = (
    "#FF00FF",  # magenta
    "#7FFF00",  # chartreuse
    "#FF8C00",  # neon orange
    "#00CED1",  # turquoise
    "#FFD700",  # gold
    "#DC143C",  # crimson
    "#00FF7F",  # spring green
)


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    text_indices = [
        i for i, n in enumerate(spec.get("nodes", [])) if n.get("type") == "text"
    ]
    if not text_indices:
        return None

    changed: list[int] = []
    for ordinal, idx in enumerate(text_indices):
        node = spec["nodes"][idx]
        new_color = CLASHING_PALETTE[ordinal % len(CLASHING_PALETTE)]
        if (node.get("color") or "").upper() == new_color.upper():
            continue
        spec["nodes"][idx] = {**node, "color": new_color}
        changed.append(idx)

    if not changed:
        return None

    return Corruption(
        spec=spec,
        description=(
            f"recolored {len(changed)} TextNodes with clashing palette "
            f"({', '.join(CLASHING_PALETTE[: min(len(changed), len(CLASHING_PALETTE))])})"
        ),
        changed_node_indices=changed,
    )


_DEF = MODE_DEFINITIONS["poor_colors"]
CORRUPTER = Corrupter(
    id="poor_colors",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
