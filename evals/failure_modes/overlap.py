"""Undesired overlap: two text elements whose bounding boxes intersect,
causing one to render on top of the other."""

from __future__ import annotations

from evals.common import make_spec, text_node
from evals.failure_modes.base import FailureMode, Variant


def _build(
    text_a: str,
    text_b: str,
    *,
    a_xywh: tuple[int, int, int, int],
    b_xywh_good: tuple[int, int, int, int],
    b_xywh_bad: tuple[int, int, int, int],
    font_size: int,
) -> Variant:
    def spec(b_xywh: tuple[int, int, int, int]) -> dict:
        ax, ay, aw, ah = a_xywh
        bx, by, bw, bh = b_xywh
        return make_spec(
            [
                text_node(text_a, ax, ay, aw, ah, font_size=font_size),
                text_node(text_b, bx, by, bw, bh, font_size=font_size),
            ]
        )

    return Variant(
        bad_spec=spec(b_xywh_bad),
        bad_svgs={},
        good_spec=spec(b_xywh_good),
        good_svgs={},
    )


def _generate() -> list[Variant]:
    return [
        _build(
            "This text overlaps",
            "another textbox",
            a_xywh=(60, 250, 420, 60),
            b_xywh_good=(60, 320, 420, 60),
            b_xywh_bad=(280, 270, 420, 60),
            font_size=34,
        ),
        _build(
            "Headline copy",
            "Subheading prose",
            a_xywh=(100, 220, 380, 56),
            b_xywh_good=(100, 290, 600, 56),
            b_xywh_bad=(220, 240, 500, 56),
            font_size=32,
        ),
        _build(
            "Top label",
            "Bottom label",
            a_xywh=(140, 230, 340, 50),
            b_xywh_good=(140, 300, 340, 50),
            b_xywh_bad=(180, 245, 400, 50),
            font_size=30,
        ),
        _build(
            "Section A",
            "Section B",
            a_xywh=(80, 240, 320, 60),
            b_xywh_good=(420, 240, 320, 60),
            b_xywh_bad=(260, 250, 360, 60),
            font_size=34,
        ),
        _build(
            "First line",
            "Second line",
            a_xywh=(100, 260, 600, 48),
            b_xywh_good=(100, 330, 600, 48),
            b_xywh_bad=(150, 275, 600, 48),
            font_size=30,
        ),
    ]


MODE = FailureMode(
    id="overlap",
    name="Undesired overlap",
    description=(
        "Two distinct text elements have intersecting bounding boxes, so one "
        "renders directly on top of the other and obscures it."
    ),
    generate=_generate,
)
