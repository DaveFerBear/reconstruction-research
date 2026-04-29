"""Inconsistency: change ONE axis on one item of a uniform-styled group.

Either the font OR the color, never both. When color is the axis, we shift
along the same hue rather than swapping to an unrelated color — visible but
realistic-looking inconsistency."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evals.common import MODE_DEFINITIONS, normalize_spec, parse_hex, to_hex
from evals.corrupters.base import Corrupter, Corruption


# Same-family swaps within either serif or sans-serif so the change is
# obvious as a typeface difference but doesn't read as a wholly alien font.
FONT_SWAP: dict[str, str] = {
    "Arial": "Helvetica",
    "Helvetica": "Arial",
    "Helvetica Neue": "Arial",
    "Roboto": "Arial",
    "Open Sans": "Arial",
    "Inter": "Arial",
    "Montserrat": "Arial",
    "Lato": "Arial",
    "Poppins": "Arial",
    "Times New Roman": "Georgia",
    "Georgia": "Times New Roman",
    "Playfair Display": "Georgia",
}


def _swap_font(font_family: str) -> str | None:
    if font_family in FONT_SWAP:
        return FONT_SWAP[font_family]
    return None


def _shift_color_subtly(color: str) -> str:
    """Push the color toward darker (if currently light) or lighter (if dark)
    along the same hue. Roughly equivalent to ±35 luminance."""
    r, g, b = parse_hex(color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    delta = -45 if luminance > 128 else +45
    return to_hex((r + delta, g + delta, b + delta))


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
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

    candidate_indices: list[int] = []
    for indices in groups.values():
        if len(indices) >= 3:
            candidate_indices = indices
            break
    if not candidate_indices:
        return None

    target_idx = candidate_indices[len(candidate_indices) // 2]
    node = spec["nodes"][target_idx]
    old_font = node.get("font_family", "Arial")
    old_color = node.get("color", "#000000")

    # Prefer font swap when the current font has a clean swap target;
    # otherwise shift the color along the same hue. Single-axis only.
    new_font = _swap_font(old_font)
    if new_font:
        spec["nodes"][target_idx] = {**node, "font_family": new_font}
        desc = (
            f"node[{target_idx}] (1 of {len(candidate_indices)} siblings): "
            f"font {old_font} -> {new_font}"
        )
    else:
        new_color = _shift_color_subtly(old_color)
        if new_color.upper() == old_color.upper():
            return None
        spec["nodes"][target_idx] = {**node, "color": new_color}
        desc = (
            f"node[{target_idx}] (1 of {len(candidate_indices)} siblings): "
            f"color {old_color} -> {new_color}"
        )
    return Corruption(spec=spec, description=desc, changed_node_indices=[target_idx])


_DEF = MODE_DEFINITIONS["inconsistency"]
CORRUPTER = Corrupter(
    id="inconsistency",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
