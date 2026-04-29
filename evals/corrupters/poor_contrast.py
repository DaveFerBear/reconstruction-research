"""Poor contrast: shift a TextNode's color toward the canvas background."""

from __future__ import annotations

from typing import Any

from evals.common import MODE_DEFINITIONS, normalize_spec, parse_hex, rgb_distance, to_hex
from evals.corrupters.base import Corrupter, Corruption


def _blend_toward(color: str, target: str, ratio: float) -> str:
    """Move `color` `ratio` of the way toward `target` in RGB."""
    r1, g1, b1 = parse_hex(color)
    r2, g2, b2 = parse_hex(target)
    blended = (
        r1 + (r2 - r1) * ratio,
        g1 + (g2 - g1) * ratio,
        b1 + (b2 - b1) * ratio,
    )
    return to_hex(blended)


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    # Skip designs whose background is an image — we can't reason about
    # contrast against an unknown texture.
    if spec.get("has_background_image"):
        return None
    bg = spec.get("background_color") or "#FFFFFF"

    # First TextNode whose color is meaningfully far from the bg.
    target_idx: int | None = None
    target_node: dict | None = None
    for i, node in enumerate(spec.get("nodes", [])):
        if node.get("type") != "text":
            continue
        if rgb_distance(node.get("color", "#000000"), bg) > 80:
            target_idx, target_node = i, node
            break
    if target_idx is None or target_node is None:
        return None

    old_color = target_node.get("color", "#000000")
    # Push the text 92% of the way to the background — visible-but-illegible.
    new_color = _blend_toward(old_color, bg, 0.92)
    spec["nodes"][target_idx] = {**target_node, "color": new_color}

    return Corruption(
        spec=spec,
        description=f"node[{target_idx}].color: {old_color} -> {new_color} on bg {bg}",
        changed_node_indices=[target_idx],
    )


_DEF = MODE_DEFINITIONS["poor_contrast"]
CORRUPTER = Corrupter(
    id="poor_contrast",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
