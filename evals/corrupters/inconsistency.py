"""Inconsistency: find a uniform-styled group of TextNodes and reskin one of them."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evals.common import MODE_DEFINITIONS, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


# Map current font to a visibly different one. Default is Comic Sans for
# anything we don't recognize — it'll always read as "wrong".
FONT_SWAP: dict[str, str] = {
    "Arial": "Georgia",
    "Helvetica": "Courier New",
    "Helvetica Neue": "Courier New",
    "Times New Roman": "Comic Sans MS",
    "Georgia": "Courier New",
    "Poppins": "Times New Roman",
    "Roboto": "Georgia",
    "Open Sans": "Times New Roman",
    "Inter": "Georgia",
    "Montserrat": "Times New Roman",
    "Lato": "Georgia",
    "Playfair Display": "Courier New",
}


def _swap_font(font_family: str) -> str:
    return FONT_SWAP.get(font_family, "Comic Sans MS")


def _swap_color(color: str) -> str:
    """Return a new color visibly different from `color`. Picks from a small
    palette, avoiding the input color itself."""
    palette = ["#C62828", "#1565C0", "#2E7D32", "#6A1B9A", "#EF6C00"]
    norm = color.upper()
    for c in palette:
        if c != norm:
            return c
    return "#C62828"


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    # Group TextNodes by exact (font_family, font_size, color, font_weight) tuple.
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, node in enumerate(spec.get("nodes", [])):
        if node.get("type") != "text":
            continue
        key = (
            node.get("font_family"),
            node.get("font_size"),
            (node.get("color") or "").upper(),
            str(node.get("font_weight")),
        )
        groups[key].append(i)

    # Need a group of 3+ uniformly-styled siblings to corrupt one.
    candidate_indices: list[int] = []
    for key, indices in groups.items():
        if len(indices) >= 3:
            candidate_indices = indices
            break
    if not candidate_indices:
        return None

    # Pick the middle item — least likely to be a category header that's "supposed"
    # to look different anyway.
    target_idx = candidate_indices[len(candidate_indices) // 2]
    node = spec["nodes"][target_idx]
    old_font = node.get("font_family", "Arial")
    old_color = node.get("color", "#000000")
    new_font = _swap_font(old_font)
    new_color = _swap_color(old_color)
    spec["nodes"][target_idx] = {**node, "font_family": new_font, "color": new_color}
    return Corruption(
        spec=spec,
        description=(
            f"node[{target_idx}] (one of {len(candidate_indices)} siblings): "
            f"font {old_font}->{new_font}, color {old_color}->{new_color}"
        ),
        changed_node_indices=[target_idx],
    )


_DEF = MODE_DEFINITIONS["inconsistency"]
CORRUPTER = Corrupter(
    id="inconsistency",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
