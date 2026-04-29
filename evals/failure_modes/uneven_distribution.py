"""Uneven distribution: a row of N elements where one has a noticeably larger
gap to its neighbor than the others have to theirs."""

from __future__ import annotations

from evals.common import make_spec, svg_node, svg_rect_outline
from evals.failure_modes.base import FailureMode, Variant


def _row_of_boxes(
    x_positions: list[int],
    *,
    y: int,
    size: int,
    stroke: str,
    stroke_width: int = 2,
) -> tuple[dict, dict[str, str]]:
    nodes: list[dict] = []
    svgs: dict[str, str] = {}
    for i, x in enumerate(x_positions):
        fn = f"svg-{i + 1}.svg"
        svgs[fn] = svg_rect_outline(stroke=stroke, stroke_width=stroke_width)
        nodes.append(svg_node(fn, x, y, size, size))
    return make_spec(nodes), svgs


def _generate() -> list[Variant]:
    # Each tuple: (size, y, stroke, stroke_width, good_xs, bad_xs)
    configs = [
        # Variant 1: 4 medium boxes, 4th shifted right by ~100px
        (80, 260, "#000000", 2, [80, 240, 400, 560], [80, 240, 400, 680]),
        # Variant 2: 4 small boxes, gap between #2 and #3 enlarged
        (60, 270, "#222222", 2, [120, 240, 360, 480], [120, 240, 420, 540]),
        # Variant 3: 5 boxes, last one too far right
        (60, 270, "#000000", 2, [80, 200, 320, 440, 560], [80, 200, 320, 440, 660]),
        # Variant 4: 4 thicker-stroke boxes, 4th shifted further
        (90, 255, "#333333", 3, [60, 230, 400, 570], [60, 230, 400, 670]),
        # Variant 5: 4 boxes with first box too far left of cluster
        (70, 265, "#111111", 2, [200, 360, 480, 600], [80, 360, 480, 600]),
    ]
    variants: list[Variant] = []
    for size, y, stroke, sw, good_xs, bad_xs in configs:
        good_spec, good_svgs = _row_of_boxes(
            good_xs, y=y, size=size, stroke=stroke, stroke_width=sw
        )
        bad_spec, bad_svgs = _row_of_boxes(
            bad_xs, y=y, size=size, stroke=stroke, stroke_width=sw
        )
        variants.append(
            Variant(bad_spec=bad_spec, bad_svgs=bad_svgs, good_spec=good_spec, good_svgs=good_svgs)
        )
    return variants


MODE = FailureMode(
    id="uneven_distribution",
    name="Uneven distribution",
    description=(
        "Multiple visually similar elements are arranged in a row or grid, but "
        "one element has a noticeably larger gap to its neighbor than the others, "
        "breaking the expected even spacing."
    ),
    generate=_generate,
)
