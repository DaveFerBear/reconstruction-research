"""Shared helpers for spec construction, SVG asset writing, and inline icons.

The eval generators emit specs in the existing `lib.types.Spec` format (no new
schema) and SVG sidecar files that `lib.render.render_image` resolves via
`asset_dir`. Everything is offline-reproducible — no FAL or image-gen calls.
"""

from __future__ import annotations

import json
from copy import deepcopy
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


# ---------------------------------------------------------------------------
# Per-mode definitions — single source of truth shared by both eval pipelines.
#
# `name` and `description` are baked into the VLM judge's system prompt; keeping
# them in one place means the synthetic eval (evals.failure_modes) and the
# real-design eval (evals.corrupters) ask identical questions.
# ---------------------------------------------------------------------------

MODE_DEFINITIONS: dict[str, dict[str, str]] = {
    "uneven_distribution": {
        "name": "Uneven distribution",
        "description": (
            "Multiple visually similar elements are arranged in a row or grid, "
            "but one element has a noticeably larger gap to its neighbor than "
            "the others, breaking the expected even spacing."
        ),
    },
    "misalignment": {
        "name": "Misalignment",
        "description": (
            "An element is nested inside a container where it visually should "
            "be centered, but it is offset to one side or corner instead of "
            "properly aligned within the parent."
        ),
    },
    "inconsistency": {
        "name": "Inconsistency",
        "description": (
            "A list or group of related items is rendered with inconsistent "
            "typography (different fonts, colors, or weights between items) "
            "when uniform styling would be expected."
        ),
    },
    "nonsensical_hierarchy": {
        "name": "Nonsensical scale/hierarchy",
        "description": (
            "The visual hierarchy is inverted: the title or heading is rendered "
            "at a smaller size than the body text, so the most important "
            "element no longer appears most prominent."
        ),
    },
    "crowding": {
        "name": "Crowding",
        "description": (
            "Multiple elements are packed directly against each other with no "
            "padding, margin, or breathing room separating them."
        ),
    },
    "semiotic_mismatch": {
        "name": "Semiotic mismatch",
        "description": (
            "An icon is paired with a text label whose semantic meaning does "
            "not match the icon — e.g., a warning triangle next to a postal "
            "address, or a phone icon next to an email address."
        ),
    },
    "overflow": {
        "name": "Undesired overflow",
        "description": (
            "Text content visibly extends beyond its intended container or "
            "wraps in an awkward way (such as breaking mid-word) because the "
            "container is too small to hold the text at the chosen font size."
        ),
    },
    "poor_contrast": {
        "name": "Poor contrast",
        "description": (
            "Text is rendered in a color so close to the background color that "
            "it is very difficult to read."
        ),
    },
    "overlap": {
        "name": "Undesired overlap",
        "description": (
            "Two distinct text elements have intersecting bounding boxes, so "
            "one renders directly on top of the other and obscures it."
        ),
    },
    "poor_colors": {
        "name": "Poor colors",
        "description": (
            "The design uses a poorly-chosen, uncoordinated, or clashing "
            "color palette. Text colors fight each other or feel garish "
            "rather than harmonizing into a cohesive scheme."
        ),
    },
    "poor_fonts": {
        "name": "Poor fonts",
        "description": (
            "The design's typeface choice is tonally inappropriate or "
            "unprofessional for the content — for example, Comic Sans, "
            "Papyrus, or a decorative script used on a serious or "
            "corporate design."
        ),
    },
}


# ---------------------------------------------------------------------------
# Bbox + color helpers used by the corrupters.
# ---------------------------------------------------------------------------


def bbox(node: dict[str, Any]) -> tuple[int, int, int, int]:
    """(x, y, w, h) for any node — TextNode, ImageNode, or SVGNode."""
    return int(node["x"]), int(node["y"]), int(node["width"]), int(node["height"])


def center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih


def parse_hex(color: str) -> tuple[int, int, int]:
    """Parse `#RRGGBB` or `#RGB` into an (r, g, b) tuple. Falls back to (0, 0, 0)."""
    s = color.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return (0, 0, 0)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return (0, 0, 0)


def to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = (max(0, min(255, int(v))) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb_distance(c1: str, c2: str) -> float:
    """Euclidean distance in RGB space; ~441 max."""
    r1, g1, b1 = parse_hex(c1)
    r2, g2, b2 = parse_hex(c2)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


def normalize_spec(spec_dict: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a spec dict and snake_case all node-field keys.

    The real specs in `datasets/specs/` use CSS-style hyphenated keys
    (`font-size`, `text-align`, …) on TextNodes. Pydantic's `populate_by_name`
    aliases let the renderer accept either form, but our corrupters work
    directly on dicts, so we normalize before mutating. Output specs are
    re-parseable by `lib.types.Spec` because the model accepts both forms.
    """
    spec = deepcopy(spec_dict)
    for node in spec.get("nodes", []):
        for key in list(node.keys()):
            if "-" in key:
                node[key.replace("-", "_")] = node.pop(key)
    return spec


# Coarse-grained categorization. The judge emits open-ended top-3 issues; an
# offline classifier maps each free-text issue to a mode_id (one of
# MODE_DEFINITIONS' keys), and we additionally roll up to one of
# {"layout", "visual"} for a coarser score.
#   layout = positional / spatial / sizing failures
#   visual = typography / color / iconography failures
SUPERCATEGORIES: dict[str, str] = {
    "uneven_distribution":   "layout",
    "misalignment":          "layout",
    "nonsensical_hierarchy": "layout",
    "crowding":              "layout",
    "overflow":              "layout",
    "overlap":               "layout",
    "inconsistency":         "visual",
    "semiotic_mismatch":     "visual",
    "poor_contrast":         "visual",
    "poor_colors":           "visual",
    "poor_fonts":            "visual",
}
