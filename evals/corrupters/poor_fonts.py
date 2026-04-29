"""Poor fonts: replace every TextNode's font_family with a tonally
inappropriate font (Comic Sans, Papyrus, Brush Script).

Distinct from `inconsistency` (which mutates ONE item out of a uniform group):
here the WHOLE design adopts an unprofessional typeface. Picks one bad font
per spec deterministically (rotated by node count) so the corpus has variety
without randomness.
"""

from __future__ import annotations

from typing import Any

from evals.common import MODE_DEFINITIONS, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


# Universally recognized "wrong for serious design" typefaces, available by
# default on macOS / Windows. Chromium on Linux may fall back to a generic
# substitute; the result still reads as visibly inappropriate.
POOR_FONTS: tuple[str, ...] = (
    "Comic Sans MS",
    "Papyrus",
    "Brush Script MT",
)


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    text_indices = [
        i for i, n in enumerate(spec.get("nodes", [])) if n.get("type") == "text"
    ]
    if not text_indices:
        return None

    # Deterministic rotation across the corpus: pick a poor font based on a
    # stable property of the spec (node count) so different designs in the
    # corpus get different bad fonts.
    poor_font = POOR_FONTS[len(spec.get("nodes", [])) % len(POOR_FONTS)]

    changed: list[int] = []
    for idx in text_indices:
        node = spec["nodes"][idx]
        if node.get("font_family") == poor_font:
            continue
        spec["nodes"][idx] = {**node, "font_family": poor_font}
        changed.append(idx)

    if not changed:
        return None

    return Corruption(
        spec=spec,
        description=f"set font_family to {poor_font!r} on {len(changed)} TextNodes",
        changed_node_indices=changed,
    )


_DEF = MODE_DEFINITIONS["poor_fonts"]
CORRUPTER = Corrupter(
    id="poor_fonts",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
