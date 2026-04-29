"""Crowding: multiple elements packed against each other with no breathing
room between them."""

from __future__ import annotations

from evals.common import MODE_DEFINITIONS, make_spec, svg_node, svg_rect_outline, text_node
from evals.failure_modes.base import FailureMode, Variant


def _build(
    letters: list[str],
    *,
    box_size: int,
    y_top: int,
    body_text: str,
    body_size: int,
    crowded: bool,
) -> tuple[dict, dict[str, str]]:
    n = len(letters)
    canvas_w = 800
    if crowded:
        # Boxes touch each other; body sits flush below with no margin.
        gap = 0
        body_offset = 0
    else:
        gap = 24
        body_offset = 36
    total_w = n * box_size + (n - 1) * gap
    start_x = (canvas_w - total_w) // 2

    svgs: dict[str, str] = {}
    nodes: list[dict] = []
    for i, letter in enumerate(letters):
        fn = f"svg-{i + 1}.svg"
        svgs[fn] = svg_rect_outline(stroke="#000000", stroke_width=3)
        x = start_x + i * (box_size + gap)
        nodes.append(svg_node(fn, x, y_top, box_size, box_size))
        nodes.append(
            text_node(
                letter,
                x,
                y_top,
                box_size,
                box_size,
                font_size=int(box_size * 0.7),
                font_weight="900",
                text_align="center",
            )
        )
    body_y = y_top + box_size + body_offset
    nodes.append(
        text_node(
            body_text,
            start_x - 50,
            body_y,
            total_w + 100,
            body_size * 3,
            font_size=body_size,
            text_align="center",
            line_height=1.0 if crowded else 1.3,
        )
    )
    return make_spec(nodes), svgs


def _generate() -> list[Variant]:
    cases = [
        (["A", "B", "C"], 130, 150, "These are 3 sections & text without any space to breathe", 26),
        (["1", "2", "3", "4"], 110, 140, "Four steps in a row with everything pressed together", 24),
        (["X", "Y"], 150, 160, "Two boxes and a label crammed against each other", 28),
        (["A", "B", "C", "D"], 100, 150, "Four-letter sequence and caption with zero gap", 22),
        (["Q", "R", "S"], 140, 130, "Three units of content abutting tightly", 24),
    ]
    variants: list[Variant] = []
    for letters, box, y_top, body, body_size in cases:
        bad_spec, bad_svgs = _build(
            letters, box_size=box, y_top=y_top, body_text=body, body_size=body_size, crowded=True
        )
        good_spec, good_svgs = _build(
            letters, box_size=box, y_top=y_top, body_text=body, body_size=body_size, crowded=False
        )
        variants.append(
            Variant(bad_spec=bad_spec, bad_svgs=bad_svgs, good_spec=good_spec, good_svgs=good_svgs)
        )
    return variants


_DEF = MODE_DEFINITIONS["crowding"]
MODE = FailureMode(
    id="crowding",
    name=_DEF["name"],
    description=_DEF["description"],
    generate=_generate,
)
