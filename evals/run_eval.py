"""Run the VLM-as-judge eval across all failure modes and models.

Two data sources:
  * `synthetic` — hand-built specs under evals/data/<mode>/<NN>/{bad,good}/render.png
  * `real`      — real Canva specs corrupted in-place: evals/data_real/<spec_id>/<mode>/render.png
                  paired against the original datasets/specs/<spec_id>/render.png

For each (mode, source, label, model) tuple it asks the VLM whether the target
failure mode is present. Reports per-(source, mode, model) precision /
recall / F1 / accuracy and writes results.json.

The expected finding is *low* judge accuracy on most modes — that is the
demonstration of VLM blindness this benchmark exists to produce.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic

from evals.failure_modes import FAILURE_MODES, FailureMode
from evals.judge import DEFAULT_MODELS, judge

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DATA_DIR = Path(__file__).parent / "data"
REAL_DATA_DIR = Path(__file__).parent / "data_real"
SOURCE_DIR = REPO_ROOT / "datasets" / "specs"
RESULTS_PATH = Path(__file__).parent / "results.json"


@dataclass
class Item:
    source: str          # "synthetic" or "real"
    mode_id: str
    variant: str         # synthetic: "01"-"05"; real: spec_id
    label: str           # "bad" or "good"
    render_path: Path

    @property
    def is_failure(self) -> bool:
        return self.label == "bad"


@dataclass
class Result:
    source: str
    mode_id: str
    variant: str
    label: str
    model: str
    expected: bool       # True iff label == "bad"
    predicted: bool
    raw: str
    input_tokens: int
    output_tokens: int


def _collect_synthetic(root: Path, only_mode: str | None) -> list[Item]:
    items: list[Item] = []
    if not root.exists():
        return items
    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir():
            continue
        if only_mode and mode_dir.name != only_mode:
            continue
        for variant_dir in sorted(mode_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            for label in ("bad", "good"):
                render = variant_dir / label / "render.png"
                if render.exists():
                    items.append(
                        Item(
                            source="synthetic",
                            mode_id=mode_dir.name,
                            variant=variant_dir.name,
                            label=label,
                            render_path=render,
                        )
                    )
    return items


def _collect_real(real_root: Path, source_root: Path, only_mode: str | None) -> list[Item]:
    items: list[Item] = []
    if not real_root.exists():
        return items
    for spec_dir in sorted(real_root.iterdir()):
        if not spec_dir.is_dir():
            continue
        original_render = source_root / spec_dir.name / "render.png"
        if not original_render.exists():
            continue
        for mode_dir in sorted(spec_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            if only_mode and mode_dir.name != only_mode:
                continue
            corrupted_render = mode_dir / "render.png"
            if not corrupted_render.exists():
                continue
            items.append(
                Item(
                    source="real",
                    mode_id=mode_dir.name,
                    variant=spec_dir.name,
                    label="bad",
                    render_path=corrupted_render,
                )
            )
            items.append(
                Item(
                    source="real",
                    mode_id=mode_dir.name,
                    variant=spec_dir.name,
                    label="good",
                    render_path=original_render,
                )
            )
    return items


def _summarize(results: list[Result]) -> dict:
    """Per-(source, model, mode) confusion matrix and metrics."""
    by_key: dict[tuple[str, str, str], dict[str, int]] = {}
    for r in results:
        key = (r.source, r.model, r.mode_id)
        bucket = by_key.setdefault(key, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
        if r.expected and r.predicted:
            bucket["tp"] += 1
        elif not r.expected and not r.predicted:
            bucket["tn"] += 1
        elif not r.expected and r.predicted:
            bucket["fp"] += 1
        else:
            bucket["fn"] += 1

    summary: dict[str, dict[str, dict[str, dict]]] = {}
    for (source, model, mode_id), c in by_key.items():
        n = c["tp"] + c["tn"] + c["fp"] + c["fn"]
        accuracy = (c["tp"] + c["tn"]) / n if n else 0.0
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        summary.setdefault(source, {}).setdefault(model, {})[mode_id] = {
            "n": n,
            "tp": c["tp"], "tn": c["tn"], "fp": c["fp"], "fn": c["fn"],
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
    return summary


def _print_table(summary_for_source: dict, *, source: str) -> None:
    print(f"\n=== {source.upper()} ===")
    models = sorted(summary_for_source)
    if not models:
        print("(no results)")
        return
    mode_ids = sorted({m for model in models for m in summary_for_source[model]})

    header = f"{'mode':<26}" + "".join(f"{m + ' acc':<22}" for m in models)
    print()
    print(header)
    print("-" * len(header))
    for mode_id in mode_ids:
        row = f"{mode_id:<26}"
        for model in models:
            metrics = summary_for_source[model].get(mode_id)
            if metrics:
                cell = f"{metrics['accuracy']:.2f} (F1 {metrics['f1']:.2f})  n={metrics['n']}"
            else:
                cell = "-"
            row += f"{cell:<22}"
        print(row)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VLM-blindness eval")
    parser.add_argument(
        "--source",
        choices=["synthetic", "real", "both"],
        default="both",
        help="Which dataset(s) to evaluate (default: both)",
    )
    parser.add_argument("--mode", help="Restrict to one failure-mode id")
    parser.add_argument(
        "--model",
        action="append",
        help=(
            "Model to evaluate (repeatable). "
            f"Default: {', '.join(DEFAULT_MODELS)}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Cap variants/specs per mode for smoke tests (per source)",
    )
    parser.add_argument(
        "--synthetic-root",
        default=str(SYNTHETIC_DATA_DIR),
        help=f"Synthetic data root (default: {SYNTHETIC_DATA_DIR})",
    )
    parser.add_argument(
        "--real-root",
        default=str(REAL_DATA_DIR),
        help=f"Real corruption data root (default: {REAL_DATA_DIR})",
    )
    parser.add_argument(
        "--source-root",
        default=str(SOURCE_DIR),
        help=f"Original specs root (for real-source goods, default: {SOURCE_DIR})",
    )
    parser.add_argument(
        "--out",
        default=str(RESULTS_PATH),
        help=f"Path for results JSON (default: {RESULTS_PATH})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent API calls (default: 8). Set to 1 for serial.",
    )
    args = parser.parse_args()

    models = tuple(args.model) if args.model else DEFAULT_MODELS

    if args.mode:
        modes_in_scope = [m for m in FAILURE_MODES if m.id == args.mode]
        if not modes_in_scope:
            raise SystemExit(f"No failure mode with id={args.mode!r}")
    else:
        modes_in_scope = FAILURE_MODES
    mode_by_id: dict[str, FailureMode] = {m.id: m for m in modes_in_scope}

    items: list[Item] = []
    if args.source in ("synthetic", "both"):
        items.extend(_collect_synthetic(Path(args.synthetic_root), args.mode))
    if args.source in ("real", "both"):
        items.extend(
            _collect_real(Path(args.real_root), Path(args.source_root), args.mode)
        )

    # Drop items whose mode_id isn't in scope (corrupters/failure_modes registries
    # may differ — e.g. semiotic_mismatch only exists in synthetic).
    items = [i for i in items if i.mode_id in mode_by_id]

    if args.limit:
        # Cap variants per (source, mode) to the first N (sorted by variant id).
        keep: dict[tuple[str, str], list[str]] = {}
        for i in items:
            key = (i.source, i.mode_id)
            keep.setdefault(key, [])
            if i.variant not in keep[key]:
                keep[key].append(i.variant)
        for key in keep:
            keep[key] = sorted(keep[key])[: args.limit]
        items = [i for i in items if i.variant in keep[(i.source, i.mode_id)]]

    if not items:
        print("No render.png files found. Did you run generate + render?", file=sys.stderr)
        sys.exit(1)

    # The Anthropic SDK's HTTP client is thread-safe; sharing one client across
    # the pool keeps connection reuse and the SDK's built-in 429/5xx retries.
    client = anthropic.Anthropic()

    jobs: list[tuple[Item, str]] = [(item, model) for item in items for model in models]
    total = len(jobs)
    print(
        f"Evaluating {len(items)} renders × {len(models)} models = {total} calls "
        f"(concurrency={args.concurrency})"
    )
    started = time.time()

    def _run(job: tuple[Item, str]) -> Result | None:
        item, model = job
        mode = mode_by_id[item.mode_id]
        try:
            v = judge(item.render_path, mode, model, client=client)
        except anthropic.APIStatusError as e:
            print(
                f"  {item.source} {model} {item.mode_id}/{item.variant}/{item.label}: "
                f"API error {e.status_code} — {e.message}",
                file=sys.stderr,
            )
            return None
        except Exception as e:
            print(
                f"  {item.source} {model} {item.mode_id}/{item.variant}/{item.label}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            traceback.print_exc()
            return None
        return Result(
            source=item.source,
            mode_id=item.mode_id,
            variant=item.variant,
            label=item.label,
            model=model,
            expected=item.is_failure,
            predicted=v.verdict,
            raw=v.raw,
            input_tokens=v.input_tokens,
            output_tokens=v.output_tokens,
        )

    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(_run, job): job for job in jobs}
        for j, future in enumerate(as_completed(futures), start=1):
            item, model = futures[future]
            result = future.result()
            if result is None:
                continue
            results.append(result)
            ok = "✓" if result.predicted == result.expected else "✗"
            print(
                f"  [{j}/{total}] {item.source} {model} "
                f"{item.mode_id}/{item.variant}/{item.label}: "
                f"{ok} (said {'YES' if result.predicted else 'NO'})"
            )

    elapsed = time.time() - started
    print(f"\nDone in {elapsed:.1f}s ({len(results)}/{total} calls succeeded)")

    summary = _summarize(results)
    for source in sorted(summary):
        _print_table(summary[source], source=source)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
