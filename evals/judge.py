"""VLM-as-judge for design failure modes.

Sends a rendered design to Claude (via the Anthropic SDK) and asks a yes/no
question scoped to one specific failure mode. The judge sees only the image —
no spec metadata, no labels — so its accuracy is interpretable as a measure
of visual perception.

A cache_control marker is set on the system prompt for hygiene, but it will
silently no-op on Claude Opus / Sonnet 4.6 because the system text is well
below the 4096-token cacheable-prefix minimum.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from evals.failure_modes import FailureMode

# Load ANTHROPIC_API_KEY from the project-root .env (matches the rest of the repo).
load_dotenv()

DEFAULT_MODELS: tuple[str, ...] = ("claude-opus-4-6", "claude-sonnet-4-6")
MAX_TOKENS = 128


@dataclass
class Verdict:
    verdict: bool  # True = VLM said YES (failure present)
    raw: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


def _encode_png(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _system_prompt(mode: FailureMode) -> list[dict]:
    text = (
        "You are evaluating a graphic design for one specific failure mode.\n\n"
        f"Failure mode: {mode.name}\n"
        f"Definition: {mode.description}\n\n"
        "Look ONLY at the image you are about to be shown. Does the design "
        "exhibit this specific failure mode?\n\n"
        "Respond with exactly one of `YES` or `NO` on the first line (no other "
        "text on that line), then one short sentence of justification on the "
        "next line."
    )
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _user_content(image_b64: str) -> list[dict]:
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_b64,
            },
        },
        {"type": "text", "text": "Evaluate this design."},
    ]


def _parse_verdict(text: str) -> bool:
    """Extract a boolean from the first non-blank token of the response."""
    for line in text.strip().splitlines():
        token = line.strip().lstrip("`").lstrip("*").strip().split()[:1]
        if not token:
            continue
        head = token[0].rstrip(".,:;").upper()
        if head.startswith("YES"):
            return True
        if head.startswith("NO"):
            return False
    raise ValueError(f"Could not parse YES/NO from response: {text!r}")


def judge(
    render_path: Path,
    mode: FailureMode,
    model: str,
    *,
    client: anthropic.Anthropic | None = None,
) -> Verdict:
    if client is None:
        client = anthropic.Anthropic()

    image_b64 = _encode_png(render_path)
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=_system_prompt(mode),
        messages=[{"role": "user", "content": _user_content(image_b64)}],
    )
    raw = "".join(block.text for block in response.content if block.type == "text")
    verdict = _parse_verdict(raw)
    usage = response.usage
    return Verdict(
        verdict=verdict,
        raw=raw,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
