"""Poor colors: replace TextNode colors with a deliberately bad two-color
combination — same pair throughout the design, alternated across TextNodes.

Distinct from `inconsistency` (one item out of a uniform group) and from
`poor_contrast` (one label vs the bg). Here the *whole design's* color
system is tonally wrong — a single ill-chosen pair that fights itself
rather than a rainbow of unrelated hues.
"""

from __future__ import annotations

from typing import Any

from evals.common import MODE_DEFINITIONS, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


# Pairs that clash on hue + saturation. Each pair stays consistent within a
# single design — different designs in the corpus get different bad pairs.
CLASHING_PAIRS: tuple[tuple[str, str], ...] = (
    ("#FF8C00", "#FF1493"),   # neon orange + hot pink
    ("#7FFF00", "#9400D3"),   # chartreuse + dark violet
    ("#FFD700", "#00CED1"),   # gold + turquoise
    ("#DC143C", "#00FF7F"),   # crimson + spring green
    ("#FF00FF", "#FFFF00"),   # magenta + canary yellow
)


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    text_indices = [
        i for i, n in enumerate(spec.get("nodes", [])) if n.get("type") == "text"
    ]
    if not text_indices:
        return None

    # Deterministic pair selection: rotate over the corpus by node count so
    # different designs get different bad pairs without randomness.
    pair = CLASHING_PAIRS[len(spec.get("nodes", [])) % len(CLASHING_PAIRS)]
    primary, secondary = pair

    changed: list[int] = []
    for ordinal, idx in enumerate(text_indices):
        node = spec["nodes"][idx]
        # Two-tone alternation. Most TextNodes get the primary; every other
        # one (by document order) gets the clashing secondary.
        new_color = primary if ordinal % 2 == 0 else secondary
        if (node.get("color") or "").upper() == new_color.upper():
            continue
        spec["nodes"][idx] = {**node, "color": new_color}
        changed.append(idx)

    if not changed:
        return None

    return Corruption(
        spec=spec,
        description=(
            f"recolored {len(changed)} TextNodes with clashing pair "
            f"{primary} / {secondary}"
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
