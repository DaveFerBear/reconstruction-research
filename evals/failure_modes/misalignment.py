"""Misalignment: an inner element is not centered within an outer container
when visual symmetry would be expected."""

from __future__ import annotations

from evals.common import make_spec, svg_node, svg_rect_outline, text_node
from evals.failure_modes.base import FailureMode, Variant


def _build(
    *,
    outer_xywh: tuple[int, int, int, int],
    inner_size: tuple[int, int],
    inner_offset_good: tuple[int, int],
    inner_offset_bad: tuple[int, int],
    label: str,
    label_size: int,
    outer_fill: str,
    outer_stroke: str,
    inner_fill: str,
    inner_stroke: str,
) -> Variant:
    ox, oy, ow, oh = outer_xywh
    iw, ih = inner_size

    def assemble(offset: tuple[int, int]) -> tuple[dict, dict[str, str]]:
        dx, dy = offset
        center_x = ox + (ow - iw) // 2
        center_y = oy + (oh - ih) // 2
        ix = center_x + dx
        iy = center_y + dy
        svgs = {
            "svg-1.svg": svg_rect_outline(stroke=outer_stroke, fill=outer_fill, stroke_width=2),
            "svg-2.svg": svg_rect_outline(stroke=inner_stroke, fill=inner_fill, stroke_width=2),
        }
        nodes = [
            svg_node("svg-1.svg", ox, oy, ow, oh),
            svg_node("svg-2.svg", ix, iy, iw, ih),
            text_node(
                label,
                ix,
                iy,
                iw,
                ih,
                font_size=label_size,
                text_align="center",
                color="#000000",
            ),
        ]
        return make_spec(nodes), svgs

    good_spec, good_svgs = assemble(inner_offset_good)
    bad_spec, bad_svgs = assemble(inner_offset_bad)
    return Variant(
        bad_spec=bad_spec, bad_svgs=bad_svgs, good_spec=good_spec, good_svgs=good_svgs
    )


def _generate() -> list[Variant]:
    return [
        _build(
            outer_xywh=(150, 200, 500, 200),
            inner_size=(200, 80),
            inner_offset_good=(0, 0),
            inner_offset_bad=(120, -30),
            label="not centered",
            label_size=22,
            outer_fill="#FFE0B2",
            outer_stroke="#F57C00",
            inner_fill="#FFFFFF",
            inner_stroke="#F57C00",
        ),
        _build(
            outer_xywh=(200, 180, 400, 240),
            inner_size=(160, 100),
            inner_offset_good=(0, 0),
            inner_offset_bad=(-80, 50),
            label="off-center",
            label_size=20,
            outer_fill="#E3F2FD",
            outer_stroke="#1976D2",
            inner_fill="#FFFFFF",
            inner_stroke="#1976D2",
        ),
        _build(
            outer_xywh=(100, 150, 600, 300),
            inner_size=(240, 120),
            inner_offset_good=(0, 0),
            inner_offset_bad=(150, 60),
            label="hello",
            label_size=24,
            outer_fill="#F3E5F5",
            outer_stroke="#7B1FA2",
            inner_fill="#FFFFFF",
            inner_stroke="#7B1FA2",
        ),
        _build(
            outer_xywh=(180, 220, 440, 160),
            inner_size=(180, 70),
            inner_offset_good=(0, 0),
            inner_offset_bad=(110, 0),
            label="aligned?",
            label_size=20,
            outer_fill="#E8F5E9",
            outer_stroke="#388E3C",
            inner_fill="#FFFFFF",
            inner_stroke="#388E3C",
        ),
        _build(
            outer_xywh=(140, 170, 520, 260),
            inner_size=(220, 90),
            inner_offset_good=(0, 0),
            inner_offset_bad=(-130, -70),
            label="left-up offset",
            label_size=20,
            outer_fill="#FFF3E0",
            outer_stroke="#E64A19",
            inner_fill="#FFFFFF",
            inner_stroke="#E64A19",
        ),
    ]


MODE = FailureMode(
    id="misalignment",
    name="Misalignment",
    description=(
        "An element is nested inside a container where it visually should be "
        "centered, but it is offset to one side or corner instead of properly "
        "aligned within the parent."
    ),
    generate=_generate,
)
