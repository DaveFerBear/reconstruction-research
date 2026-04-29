"""Map free-text VLM-emitted design issues onto our 11-mode taxonomy.

The judge emits open-ended issue strings; this module classifies each string
into (mode_id, supercategory) using `claude-haiku-4-5` via litellm. Results
are persistently cached at `evals/classifier_cache.json` keyed by SHA256 of
the trimmed lowercased issue text — re-runs are nearly free for issues we've
seen before.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv
from tqdm import tqdm

from evals.common import MODE_DEFINITIONS, SUPERCATEGORIES

load_dotenv()

CLASSIFIER_MODEL = "claude-haiku-4-5"
CACHE_PATH = Path(__file__).parent / "classifier_cache.json"
MAX_TOKENS = 64


@dataclass(frozen=True)
class IssueCategory:
    mode_id: str | None        # one of MODE_DEFINITIONS keys, or None
    supercategory: str | None  # "layout" | "visual" | None


def _build_prompt() -> str:
    parts = [
        "You are a strict classifier. You will be given one short design-review "
        "note and must categorize it into a fixed taxonomy.",
        "",
        "## Specific failure modes",
        "Pick the closest matching mode, or null if no specific mode applies.",
        "",
    ]
    for mode_id, defn in MODE_DEFINITIONS.items():
        parts.append(f"- `{mode_id}`: {defn['description']}")

    layout_modes = [m for m, s in SUPERCATEGORIES.items() if s == "layout"]
    visual_modes = [m for m, s in SUPERCATEGORIES.items() if s == "visual"]
    parts += [
        "",
        "## Supercategories",
        f"- `layout` — positional / spatial / sizing failures: {', '.join(layout_modes)}.",
        f"- `visual` — typography / color / iconography failures: {', '.join(visual_modes)}.",
        "",
        "## Output format",
        'Output ONLY a JSON object with two keys, no commentary or markdown:',
        '  {"mode": "<mode_id or null>", "supercategory": "<\\"layout\\"|\\"visual\\"|null>"}',
        "",
        "Rules:",
        "- If the note matches a specific mode, set both `mode` and `supercategory`.",
        "- If the note clearly fits a supercategory but no specific mode applies, "
        'set mode=null and supercategory="layout" or "visual".',
        "- If the note is generic praise, generic vague feedback, or doesn't fit "
        "any of the above, output {\"mode\": null, \"supercategory\": null}.",
    ]
    return "\n".join(parts)


_PROMPT = _build_prompt()


def _key(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:16]


class _Cache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"warning: classifier cache unreadable ({e}); starting fresh", file=sys.stderr)
                self._data = {}

    def get(self, text: str) -> dict[str, Any] | None:
        with self._lock:
            return self._data.get(_key(text))

    def set(self, text: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._data[_key(text)] = {"text": text, **result}

    def save(self) -> None:
        with self._lock:
            self.path.write_text(
                json.dumps(self._data, indent=2, sort_keys=True),
                encoding="utf-8",
            )


def _parse_classification(text: str) -> dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    s = s.strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", s, re.DOTALL)
        if not match:
            return {"mode": None, "supercategory": None}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"mode": None, "supercategory": None}
    if not isinstance(data, dict):
        return {"mode": None, "supercategory": None}

    raw_mode = data.get("mode")
    raw_super = data.get("supercategory")
    mode: str | None = None
    if isinstance(raw_mode, str) and raw_mode.lower() in MODE_DEFINITIONS:
        mode = raw_mode.lower()
    super_cat: str | None = None
    if isinstance(raw_super, str) and raw_super.lower() in ("layout", "visual"):
        super_cat = raw_super.lower()
    # Reconcile: if mode is set, the canonical supercategory wins.
    if mode and SUPERCATEGORIES.get(mode):
        super_cat = SUPERCATEGORIES[mode]
    return {"mode": mode, "supercategory": super_cat}


def _classify_one(text: str, cache: _Cache | None) -> IssueCategory:
    if cache is not None:
        hit = cache.get(text)
        if hit is not None:
            return IssueCategory(mode_id=hit.get("mode"), supercategory=hit.get("supercategory"))

    response = litellm.completion(
        model=CLASSIFIER_MODEL,
        messages=[
            {"role": "system", "content": _PROMPT},
            {"role": "user", "content": f"Note: {text!r}"},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0,
    )
    raw = response.choices[0].message.content or ""
    result = _parse_classification(raw)
    if cache is not None:
        cache.set(text, result)
    return IssueCategory(mode_id=result["mode"], supercategory=result["supercategory"])


def classify(text: str) -> IssueCategory:
    cache = _Cache(CACHE_PATH)
    out = _classify_one(text, cache)
    cache.save()
    return out


def classify_batch(
    texts: list[str],
    *,
    concurrency: int = 16,
    progress: bool = True,
) -> list[IssueCategory]:
    """Classify each item in `texts`. Returns aligned results (one per input)."""
    cache = _Cache(CACHE_PATH)
    unique: list[str] = []
    seen: set[str] = set()
    for t in texts:
        if t and t not in seen:
            unique.append(t)
            seen.add(t)

    cached_count = sum(1 for t in unique if cache.get(t) is not None)
    new_count = len(unique) - cached_count
    if progress:
        print(
            f"classifier: {len(texts)} input issues, {len(unique)} unique, "
            f"{cached_count} in cache, {new_count} new calls"
        )

    results: dict[str, IssueCategory] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(_classify_one, t, cache): t for t in unique}
        bar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="classifying",
            unit="issue",
            disable=not progress,
        )
        for future in bar:
            t = futures[future]
            try:
                results[t] = future.result()
            except Exception as e:
                tqdm.write(
                    f"  classifier error on {t!r}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                results[t] = IssueCategory(None, None)

    cache.save()
    if progress:
        print(f"classifier: cache now has {len(cache._data)} entries")
    return [results.get(t, IssueCategory(None, None)) for t in texts]
