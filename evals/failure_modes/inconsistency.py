"""Inconsistency: a list/group of items where each item uses a different
font, color, or weight where uniform styling is expected."""

from __future__ import annotations

from evals.common import MODE_DEFINITIONS, make_spec, text_node
from evals.failure_modes.base import FailureMode, Variant


def _bullet_list(
    items: list[tuple[str, str, str, int]],
    *,
    x: int,
    y: int,
    width: int,
    line_height_px: int,
) -> dict:
    """items: list of (text, font_family, color, font_weight as int)."""
    nodes = []
    for i, (text, family, color, weight) in enumerate(items):
        nodes.append(
            text_node(
                f"•  {text}",
                x,
                y + i * line_height_px,
                width,
                line_height_px,
                font_size=28,
                font_family=family,
                color=color,
                font_weight=str(weight),
            )
        )
    return make_spec(nodes)


def _generate() -> list[Variant]:
    cases: list[tuple[list[str], list[tuple[str, str, int]], list[tuple[str, str, int]]]] = [
        # (items, good_styles[(family, color, weight)], bad_styles)
        (
            ["First point", "Second point", "Third point"],
            [("Arial", "#000000", 400)] * 3,
            [
                ("Arial", "#000000", 400),
                ("Arial", "#000000", 400),
                ("Playfair Display", "#000000", 700),
            ],
        ),
        (
            ["Alpha item", "Beta item", "Gamma item"],
            [("Helvetica", "#222222", 400)] * 3,
            [
                ("Helvetica", "#222222", 400),
                ("Courier New", "#B71C1C", 400),
                ("Helvetica", "#222222", 400),
            ],
        ),
        (
            ["Read the brief", "Sketch options", "Pick a direction", "Refine"],
            [("Arial", "#1A1A1A", 400)] * 4,
            [
                ("Arial", "#1A1A1A", 400),
                ("Georgia", "#1A1A1A", 400),
                ("Arial", "#1A1A1A", 700),
                ("Arial", "#1A1A1A", 400),
            ],
        ),
        (
            ["One", "Two", "Three"],
            [("Poppins", "#111111", 500)] * 3,
            [
                ("Poppins", "#111111", 500),
                ("Poppins", "#111111", 500),
                ("Times New Roman", "#0D47A1", 700),
            ],
        ),
        (
            ["Plan", "Build", "Ship"],
            [("Arial", "#000000", 400)] * 3,
            [
                ("Arial", "#000000", 400),
                ("Arial", "#777777", 400),
                ("Arial", "#000000", 800),
            ],
        ),
    ]

    variants: list[Variant] = []
    x, y, width, line_h = 120, 140, 560, 70
    for items, good_styles, bad_styles in cases:
        good = _bullet_list(
            [(t, fam, col, w) for t, (fam, col, w) in zip(items, good_styles)],
            x=x, y=y, width=width, line_height_px=line_h,
        )
        bad = _bullet_list(
            [(t, fam, col, w) for t, (fam, col, w) in zip(items, bad_styles)],
            x=x, y=y, width=width, line_height_px=line_h,
        )
        variants.append(Variant(bad_spec=bad, bad_svgs={}, good_spec=good, good_svgs={}))
    return variants


_DEF = MODE_DEFINITIONS["inconsistency"]
MODE = FailureMode(
    id="inconsistency",
    name=_DEF["name"],
    description=_DEF["description"],
    generate=_generate,
)
