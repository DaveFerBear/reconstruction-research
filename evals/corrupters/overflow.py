"""Overflow: lengthen a TextNode's text so it can't fit in its container.

Just shrinking the container width sometimes silently no-ops if the original
text already had a lot of slack. Instead, we *append* to the text — appending
guarantees overflow regardless of the original container's slack.
"""

from __future__ import annotations

from typing import Any

from evals.common import MODE_DEFINITIONS, normalize_spec
from evals.corrupters.base import Corrupter, Corruption


# Padding suffix appended in 4-word chunks until the text overflows the
# container at the original font size by a comfortable margin.
_PAD_PHRASE = (
    "with additional descriptive copy that significantly exceeds the "
    "available container width and continues for several lines beyond what "
    "the layout was designed to accommodate"
)


def _apply(spec_dict: dict[str, Any]) -> Corruption | None:
    spec = normalize_spec(spec_dict)
    candidates: list[tuple[int, dict]] = []
    for i, node in enumerate(spec.get("nodes", [])):
        if node.get("type") != "text":
            continue
        text = node.get("text") or ""
        if len(text.strip()) < 2:
            continue
        if int(node.get("width", 0)) < 60 or int(node.get("font_size", 0)) < 12:
            continue
        candidates.append((i, node))
    if not candidates:
        return None

    # Take the first TextNode in document order — keeps the corpus
    # deterministic and tends to pick the most visually prominent text.
    idx, node = candidates[0]
    old_text = node["text"]
    width = int(node["width"])
    height = int(node["height"])
    font_size = float(node["font_size"])
    line_height = float(node.get("line_height") or 1.2)

    # Approximate target string length such that the text would need ~3x
    # the available container area to fit. Empirically each character takes
    # ~0.55 * font_size px of horizontal space.
    char_width_px = font_size * 0.55
    line_height_px = font_size * line_height
    chars_per_line = max(1, int(width / char_width_px))
    lines_available = max(1, int(height / line_height_px))
    target_chars = chars_per_line * lines_available * 3

    new_text = old_text
    while len(new_text) < target_chars:
        new_text = f"{new_text} {_PAD_PHRASE}"
    # Hard cap so we don't produce absurd 10k-char strings on tiny containers.
    new_text = new_text[: max(target_chars, len(old_text) + 80)]

    spec["nodes"][idx] = {**node, "text": new_text}

    return Corruption(
        spec=spec,
        description=(
            f"node[{idx}].text: {len(old_text)} chars -> {len(new_text)} chars "
            f"(container {width}x{height} @ {int(font_size)}px)"
        ),
        changed_node_indices=[idx],
    )


_DEF = MODE_DEFINITIONS["overflow"]
CORRUPTER = Corrupter(
    id="overflow",
    name=_DEF["name"],
    description=_DEF["description"],
    apply=_apply,
)
