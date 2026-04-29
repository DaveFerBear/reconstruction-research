"""Open-ended VLM judge for design issues.

Each call asks the VLM to identify the THREE biggest design issues it sees
in the rendered image, as a JSON array of free-text strings. No taxonomy is
shown to the VLM — the offline classifier (`evals.classify`) maps the
free-text issues onto our 11-mode taxonomy after the fact.

Multi-provider via `litellm`: same call site for `claude-opus-4-6`,
`claude-sonnet-4-6`, and `gpt-4o`. API keys (`ANTHROPIC_API_KEY` and
`OPENAI_API_KEY`) come from the project-root `.env`.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

load_dotenv()

# Drop provider-unsupported params instead of erroring. Specifically: GPT-5
# rejects `temperature=0` (only temperature=1 is supported). Without this
# flag the run dies on the first gpt-5 call. Other models still see
# temperature=0 and behave deterministically.
litellm.drop_params = True

DEFAULT_MODELS: tuple[str, ...] = (
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "gpt-4o",
    "ollama/qwen3-vl:4b",
)
# Generous output budget. Reasoning models (gpt-5, qwen3-vl-thinking)
# consume a chunk of this on internal reasoning before producing visible
# `content`; with a tight cap (e.g. 300) the JSON output gets truncated
# and we silently parse an empty issues list. Non-reasoning models still
# only emit ~50-150 tokens for the actual answer, so the cost overhead
# of a high cap is negligible.
MAX_TOKENS = 2048

OLLAMA_PREFIX = "ollama/"

_SYSTEM_PROMPT = (
    "You are a senior graphic designer reviewing a finished design. Look at "
    "the image and identify the THREE biggest design issues you see, ranked "
    "from most to least serious. If you don't see three real issues, list "
    "fewer; if you see no real issues, output an empty array.\n\n"
    "Be concrete and specific — name what is wrong, not just that something "
    "is wrong.\n\n"
    "Output ONLY a JSON array of short strings. Example:\n"
    '["text overflows the container at the top right", '
    '"label and address have mismatched icons", '
    '"the body copy is larger than the headline"]\n\n'
    "Do not include any other commentary, markdown code fences, or "
    "explanation outside the JSON array."
)


@dataclass
class IssueSet:
    issues: list[str]
    raw: str
    model: str
    input_tokens: int
    output_tokens: int
    parse_error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _encode_png(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _data_url(image_b64: str) -> str:
    return f"data:image/png;base64,{image_b64}"


def _parse_issues(text: str) -> tuple[list[str], str | None]:
    """Extract up to 3 issue strings from a VLM response.

    Returns (issues, parse_error). On any JSON failure, falls back to a
    permissive regex search for a top-level array. On total failure returns
    ([], "<reason>") so the caller can still record the raw text."""
    s = text.strip()
    # Strip markdown fences if the model added them despite instructions.
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    s = s.strip()

    def _coerce(data: Any) -> list[str]:
        if not isinstance(data, list):
            return []
        out: list[str] = []
        for item in data:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                # Some models emit [{"issue": "..."}] despite our instructions.
                for key in ("issue", "description", "text"):
                    if key in item and isinstance(item[key], str):
                        out.append(item[key].strip())
                        break
        return out[:3]

    try:
        return _coerce(json.loads(s)), None
    except json.JSONDecodeError:
        pass
    # Fallback: find the first balanced [...] block.
    match = re.search(r"\[.*\]", s, re.DOTALL)
    if match:
        try:
            return _coerce(json.loads(match.group(0))), None
        except json.JSONDecodeError as e:
            return [], f"json fallback failed: {e}"
    return [], "no JSON array found in response"


def _judge_litellm(render_path: Path, model: str) -> IssueSet:
    image_b64 = _encode_png(render_path)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_url(image_b64)}},
                {"type": "text", "text": "What are the top design issues with this image?"},
            ],
        },
    ]
    response = litellm.completion(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0,
    )
    raw = response.choices[0].message.content or ""
    issues, parse_error = _parse_issues(raw)
    usage = getattr(response, "usage", None)
    return IssueSet(
        issues=issues,
        raw=raw,
        model=model,
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        parse_error=parse_error,
    )


def _judge_ollama(render_path: Path, model: str, *, full_name: str) -> IssueSet:
    """Local Ollama vision model (e.g. qwen3-vl:4b).

    Ollama's chat API takes image *paths* (or bytes) on a top-level `images`
    field of the user message, not as content blocks — see notebooks/qwen3.ipynb.
    Thinking output (for reasoning models like qwen3-vl) lands on
    `response.message.thinking` and is NOT included in `.content`.
    """
    try:
        from ollama import chat as ollama_chat  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "The `ollama` Python package is required to evaluate ollama/* models.\n"
            "Install it with `pip install ollama` (or `pip install -r requirements.txt`),\n"
            "and ensure the Ollama daemon is running (`ollama serve`)."
        ) from e

    response = ollama_chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "What are the top design issues with this image?",
                "images": [str(render_path.resolve())],
            },
        ],
        # think=False skips the internal <think> block so all output budget
        # goes to the JSON answer (qwen3-vl is a reasoning model by default).
        think=False,
        options={"temperature": 0, "num_predict": MAX_TOKENS},
    )
    raw = (response.message.content or "").strip()
    issues, parse_error = _parse_issues(raw)
    return IssueSet(
        issues=issues,
        raw=raw,
        model=full_name,
        input_tokens=int(getattr(response, "prompt_eval_count", 0) or 0),
        output_tokens=int(getattr(response, "eval_count", 0) or 0),
        parse_error=parse_error,
    )


def judge(render_path: Path, model: str) -> IssueSet:
    if model.startswith(OLLAMA_PREFIX):
        return _judge_ollama(
            render_path,
            model.removeprefix(OLLAMA_PREFIX),
            full_name=model,
        )
    return _judge_litellm(render_path, model)
