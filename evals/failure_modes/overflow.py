"""Undesired overflow: text whose required width or height exceeds its
container, causing visible overflow or awkward wrapping."""

from __future__ import annotations

from evals.common import MODE_DEFINITIONS, make_spec, svg_node, svg_rect_outline, text_node
from evals.failure_modes.base import FailureMode, Variant


def _build(
    text: str,
    *,
    container_xywh: tuple[int, int, int, int],
    font_size: int,
    overflow: bool,
) -> tuple[dict, dict[str, str]]:
    cx, cy, cw, ch = container_xywh
    if overflow:
        # Text container is half-width; CSS won't clip so the text bleeds out.
        text_w = cw // 2
    else:
        # Text container fits the bounding box.
        text_w = cw - 24

    svgs = {"svg-1.svg": svg_rect_outline(stroke="#000000", stroke_width=2)}
    nodes = [
        svg_node("svg-1.svg", cx, cy, cw, ch),
        text_node(
            text,
            cx + 12,
            cy + 12,
            text_w,
            ch - 24,
            font_size=font_size,
            line_height=1.2,
        ),
    ]
    return make_spec(nodes), svgs


def _generate() -> list[Variant]:
    cases = [
        ("This text unexpectedly overflows", (100, 240, 600, 130), 38),
        ("A surprisingly long sentence that does not fit", (120, 220, 560, 130), 32),
        ("Headline that exceeds the width", (90, 230, 620, 110), 36),
        ("Description copy with too many characters for the box", (100, 200, 600, 160), 30),
        ("Tagline overflowing the available space", (110, 250, 580, 120), 34),
    ]
    variants: list[Variant] = []
    for text, container, fs in cases:
        bad_spec, bad_svgs = _build(text, container_xywh=container, font_size=fs, overflow=True)
        good_spec, good_svgs = _build(text, container_xywh=container, font_size=fs, overflow=False)
        variants.append(
            Variant(bad_spec=bad_spec, bad_svgs=bad_svgs, good_spec=good_spec, good_svgs=good_svgs)
        )
    return variants


_DEF = MODE_DEFINITIONS["overflow"]
MODE = FailureMode(
    id="overflow",
    name=_DEF["name"],
    description=_DEF["description"],
    generate=_generate,
)
