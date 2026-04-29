"""Shared helpers for spec construction, SVG asset writing, and inline icons.

The eval generators emit specs in the existing `lib.types.Spec` format (no new
schema) and SVG sidecar files that `lib.render.render_image` resolves via
`asset_dir`. Everything is offline-reproducible — no FAL or image-gen calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CANVAS_W = 800
CANVAS_H = 600


def make_spec(
    nodes: list[dict[str, Any]],
    *,
    canvas_width: int = CANVAS_W,
    canvas_height: int = CANVAS_H,
    background_color: str = "#FFFFFF",
) -> dict[str, Any]:
    return {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "background_color": background_color,
        "has_background_image": False,
        "background_image_description": "",
        "nodes": nodes,
    }


def text_node(
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    font_size: int = 24,
    font_family: str = "Arial",
    color: str = "#000000",
    text_align: str = "left",
    font_weight: str = "normal",
    line_height: float = 1.2,
) -> dict[str, Any]:
    return {
        "type": "text",
        "text": text,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "rotation": 0,
        "opacity": 1,
        "font_family": font_family,
        "font_size": font_size,
        "color": color,
        "text_align": text_align,
        "font_weight": font_weight,
        "font_style": "normal",
        "font_stretch": "normal",
        "text_decoration": "none",
        "text_transform": "none",
        "line_height": line_height,
    }


def svg_node(
    filename: str,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    description: str = "",
) -> dict[str, Any]:
    return {
        "type": "svg",
        "svg_description": description,
        "filename": filename,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "rotation": 0,
        "opacity": 1,
    }


def svg_rect_outline(
    stroke: str = "#000000",
    stroke_width: int = 2,
    fill: str = "none",
) -> str:
    """SVG rectangle outline that scales to fill its container.

    `vector-effect="non-scaling-stroke"` keeps the stroke a constant width
    regardless of how the parent <div> stretches the SVG.
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
        f'preserveAspectRatio="none">'
        f'<rect x="0" y="0" width="100" height="100" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" '
        f'vector-effect="non-scaling-stroke"/>'
        f'</svg>'
    )


def svg_rect_filled(fill: str = "#000000") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
        f'preserveAspectRatio="none">'
        f'<rect x="0" y="0" width="100" height="100" fill="{fill}"/>'
        f'</svg>'
    )


# Material-Design-style monochrome icons. viewBox 0 0 24 24, fill via attribute
# so the renderer can scale them into any container.

ICON_PHONE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#1976D2">'
    '<path d="M6.62 10.79c1.44 2.83 3.76 5.14 6.59 6.59l2.2-2.2c.27-.27.67-.36 '
    '1.02-.24 1.12.37 2.33.57 3.57.57.55 0 1 .45 1 1V20c0 .55-.45 1-1 1-9.39 '
    '0-17-7.61-17-17 0-.55.45-1 1-1h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 '
    '1.02l-2.2 2.2z"/>'
    '</svg>'
)

ICON_ENVELOPE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#1976D2">'
    '<path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 '
    '2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/>'
    '</svg>'
)

ICON_WARNING = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#F57C00">'
    '<path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>'
    '</svg>'
)

ICON_PIN = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#1976D2">'
    '<path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 '
    '9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>'
    '</svg>'
)


def write_spec_dir(out_dir: Path, spec: dict[str, Any], svgs: dict[str, str]) -> None:
    """Write spec.json plus any svg-N.svg sidecars into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    for filename, content in svgs.items():
        (out_dir / filename).write_text(content, encoding="utf-8")
