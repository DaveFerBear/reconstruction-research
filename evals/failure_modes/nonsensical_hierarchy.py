"""Nonsensical scale/hierarchy: title is rendered smaller than body text,
inverting the visual importance hierarchy."""

from __future__ import annotations

from evals.common import MODE_DEFINITIONS, make_spec, text_node
from evals.failure_modes.base import FailureMode, Variant


def _layout(
    title: str,
    body: str,
    *,
    title_size: int,
    body_size: int,
) -> dict:
    nodes = [
        text_node(title, 100, 180, 600, title_size + 20,
                  font_size=title_size, font_weight="700", text_align="center"),
        text_node(body, 100, 280, 600, body_size * 4,
                  font_size=body_size, font_weight="400", text_align="center",
                  line_height=1.3),
    ]
    return make_spec(nodes)


def _generate() -> list[Variant]:
    cases = [
        ("This is a title", "And this is the first body paragraph.", 18, 42),
        ("Quarterly report", "Revenue grew by twelve percent this quarter.", 16, 36),
        ("About us", "We design tools for thoughtful teams.", 20, 40),
        ("Contact", "Reach our team at hello@example.com any weekday.", 18, 34),
        ("Conclusion", "The experiment failed to reject the null hypothesis.", 16, 38),
    ]
    variants: list[Variant] = []
    for title, body, small, large in cases:
        # Bad: title small, body large
        bad = _layout(title, body, title_size=small, body_size=large)
        # Good: title large, body small
        good = _layout(title, body, title_size=large, body_size=small)
        variants.append(Variant(bad_spec=bad, bad_svgs={}, good_spec=good, good_svgs={}))
    return variants


_DEF = MODE_DEFINITIONS["nonsensical_hierarchy"]
MODE = FailureMode(
    id="nonsensical_hierarchy",
    name=_DEF["name"],
    description=_DEF["description"],
    generate=_generate,
)
