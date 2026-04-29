"""Semiotic mismatch: an icon is paired with a text label whose meaning does
not match the icon (e.g. a warning glyph next to a postal address)."""

from __future__ import annotations

from evals.common import (
    ICON_ENVELOPE,
    ICON_PHONE,
    ICON_PIN,
    ICON_WARNING,
    MODE_DEFINITIONS,
    make_spec,
    svg_node,
    text_node,
)
from evals.failure_modes.base import FailureMode, Variant


def _row(
    rows: list[tuple[str, str]],  # (icon_svg, label_text)
    *,
    x: int,
    y: int,
    icon_size: int,
    line_height_px: int,
    label_offset: int,
) -> tuple[dict, dict[str, str]]:
    nodes: list[dict] = []
    svgs: dict[str, str] = {}
    for i, (icon, label) in enumerate(rows):
        fn = f"svg-{i + 1}.svg"
        svgs[fn] = icon
        row_y = y + i * line_height_px
        nodes.append(svg_node(fn, x, row_y, icon_size, icon_size))
        nodes.append(
            text_node(
                label,
                x + icon_size + label_offset,
                row_y,
                500,
                icon_size,
                font_size=24,
            )
        )
    return make_spec(nodes), svgs


def _generate() -> list[Variant]:
    cases: list[tuple[
        list[tuple[str, str]],  # good rows
        list[tuple[str, str]],  # bad rows (one icon swapped to a wrong meaning)
    ]] = [
        (
            [
                (ICON_PHONE, "(123) 456 7890"),
                (ICON_ENVELOPE, "contact@email.com"),
                (ICON_PIN, "10 Main St. Anytown USA"),
            ],
            [
                (ICON_PHONE, "(123) 456 7890"),
                (ICON_ENVELOPE, "contact@email.com"),
                (ICON_WARNING, "10 Main St. Anytown USA"),
            ],
        ),
        (
            [
                (ICON_ENVELOPE, "support@company.io"),
                (ICON_PHONE, "+1 (555) 010-2020"),
                (ICON_PIN, "500 Market St, San Francisco"),
            ],
            [
                (ICON_PIN, "support@company.io"),
                (ICON_PHONE, "+1 (555) 010-2020"),
                (ICON_PIN, "500 Market St, San Francisco"),
            ],
        ),
        (
            [
                (ICON_PHONE, "+44 20 7946 0123"),
                (ICON_ENVELOPE, "hello@studio.uk"),
                (ICON_PIN, "12 Old Street, London"),
            ],
            [
                (ICON_WARNING, "+44 20 7946 0123"),
                (ICON_ENVELOPE, "hello@studio.uk"),
                (ICON_PIN, "12 Old Street, London"),
            ],
        ),
        (
            [
                (ICON_PIN, "1 Park Ave, NY"),
                (ICON_ENVELOPE, "office@firm.com"),
                (ICON_PHONE, "(212) 555 0199"),
            ],
            [
                (ICON_PIN, "1 Park Ave, NY"),
                (ICON_PHONE, "office@firm.com"),
                (ICON_PHONE, "(212) 555 0199"),
            ],
        ),
        (
            [
                (ICON_PHONE, "555 0100"),
                (ICON_ENVELOPE, "team@example.org"),
                (ICON_PIN, "PO Box 100, Boulder CO"),
            ],
            [
                (ICON_PHONE, "555 0100"),
                (ICON_WARNING, "team@example.org"),
                (ICON_PIN, "PO Box 100, Boulder CO"),
            ],
        ),
    ]

    variants: list[Variant] = []
    for good_rows, bad_rows in cases:
        good_spec, good_svgs = _row(
            good_rows, x=140, y=180, icon_size=42, line_height_px=80, label_offset=18
        )
        bad_spec, bad_svgs = _row(
            bad_rows, x=140, y=180, icon_size=42, line_height_px=80, label_offset=18
        )
        variants.append(
            Variant(bad_spec=bad_spec, bad_svgs=bad_svgs, good_spec=good_spec, good_svgs=good_svgs)
        )
    return variants


_DEF = MODE_DEFINITIONS["semiotic_mismatch"]
MODE = FailureMode(
    id="semiotic_mismatch",
    name=_DEF["name"],
    description=_DEF["description"],
    generate=_generate,
)
