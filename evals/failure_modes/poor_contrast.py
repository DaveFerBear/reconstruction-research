"""Poor contrast: text rendered in a color very close to the background,
making it hard to read."""

from __future__ import annotations

from evals.common import MODE_DEFINITIONS, make_spec, text_node
from evals.failure_modes.base import FailureMode, Variant


def _build(
    text: str,
    *,
    bg: str,
    text_color_bad: str,
    text_color_good: str,
    font_size: int,
) -> tuple[dict, dict]:
    def spec(color: str) -> dict:
        s = make_spec(
            [
                text_node(
                    text,
                    80,
                    250,
                    640,
                    font_size + 20,
                    font_size=font_size,
                    color=color,
                    text_align="center",
                )
            ],
            background_color=bg,
        )
        return s

    return spec(text_color_bad), spec(text_color_good)


def _generate() -> list[Variant]:
    cases = [
        ("This text lacks contrast", "#FFFFFF", "#E8DDF2", "#1A1A1A", 38),
        ("Light grey on white", "#FFFFFF", "#EDEDED", "#111111", 36),
        ("Pale yellow on cream", "#FFF8E1", "#F5EFC8", "#0D47A1", 34),
        ("Faded teal on mint", "#E0F2F1", "#B2DFDB", "#004D40", 36),
        ("Almost invisible", "#FFFFFF", "#F0F0F0", "#212121", 40),
    ]
    variants: list[Variant] = []
    for text, bg, bad_color, good_color, fs in cases:
        bad, good = _build(
            text,
            bg=bg,
            text_color_bad=bad_color,
            text_color_good=good_color,
            font_size=fs,
        )
        variants.append(Variant(bad_spec=bad, bad_svgs={}, good_spec=good, good_svgs={}))
    return variants


_DEF = MODE_DEFINITIONS["poor_contrast"]
MODE = FailureMode(
    id="poor_contrast",
    name=_DEF["name"],
    description=_DEF["description"],
    generate=_generate,
)
