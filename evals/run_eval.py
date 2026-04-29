"""Run the VLM-as-judge eval across all failure modes and models.

For each (mode, variant, label, model) it asks the VLM whether the target
failure mode is present in the rendered image. Reports per-(mode, model)
precision / recall / F1 / accuracy, and writes results.json.

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

DATA_DIR = Path(__file__).parent / "data"
RESULTS_PATH = DATA_DIR / "results.json"


@dataclass
class Item:
    mode_id: str
    variant: str  # "01"..."05"
    label: str  # "bad" or "good"
    render_path: Path

    @property
    def is_failure(self) -> bool:
        return self.label == "bad"


@dataclass
class Result:
    mode_id: str
    variant: str
    label: str
    model: str
    expected: bool  # True iff label == "bad"
    predicted: bool
    raw: str
    input_tokens: int
    output_tokens: int


def _collect_items(root: Path, only_mode: str | None) -> list[Item]:
    items: list[Item] = []
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
                            mode_id=mode_dir.name,
                            variant=variant_dir.name,
                            label=label,
                            render_path=render,
                        )
                    )
    return items


def _summarize(results: list[Result]) -> dict:
    """Per-(mode, model) confusion matrix and metrics."""
    by_key: dict[tuple[str, str], dict[str, int]] = {}
    for r in results:
        key = (r.mode_id, r.model)
        bucket = by_key.setdefault(key, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
        if r.expected and r.predicted:
            bucket["tp"] += 1
        elif not r.expected and not r.predicted:
            bucket["tn"] += 1
        elif not r.expected and r.predicted:
            bucket["fp"] += 1
        else:
            bucket["fn"] += 1

    summary: dict[str, dict[str, dict]] = {}
    for (mode_id, model), c in by_key.items():
        n = c["tp"] + c["tn"] + c["fp"] + c["fn"]
        accuracy = (c["tp"] + c["tn"]) / n if n else 0.0
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        summary.setdefault(model, {})[mode_id] = {
            "n": n,
            "tp": c["tp"], "tn": c["tn"], "fp": c["fp"], "fn": c["fn"],
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
    return summary


def _print_table(summary: dict) -> None:
    models = sorted(summary)
    if not models:
        print("(no results)")
        return
    mode_ids = sorted({m for model in models for m in summary[model]})

    header = f"{'mode':<26}" + "".join(f"{m + ' acc':<22}" for m in models)
    print()
    print(header)
    print("-" * len(header))
    for mode_id in mode_ids:
        row = f"{mode_id:<26}"
        for model in models:
            metrics = summary[model].get(mode_id)
            if metrics:
                cell = f"{metrics['accuracy']:.2f} (F1 {metrics['f1']:.2f})"
            else:
                cell = "-"
            row += f"{cell:<22}"
        print(row)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VLM-blindness eval")
    parser.add_argument("--mode", help="Restrict to one failure-mode id")
    parser.add_argument(
        "--model",
        action="append",
        help=(
            "Model to evaluate (repeatable). "
            f"Default: {', '.join(DEFAULT_MODELS)}"
        ),
    )
    parser.add_argument("--limit", type=int, help="Cap variants per mode for smoke tests")
    parser.add_argument("--root", default=str(DATA_DIR), help="Data root")
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

    root = Path(args.root)
    if not root.exists():
        print(f"No data at {root}; run generate + render first.", file=sys.stderr)
        sys.exit(1)

    models = tuple(args.model) if args.model else DEFAULT_MODELS

    if args.mode:
        modes_in_scope = [m for m in FAILURE_MODES if m.id == args.mode]
        if not modes_in_scope:
            raise SystemExit(f"No failure mode with id={args.mode!r}")
    else:
        modes_in_scope = FAILURE_MODES
    mode_by_id: dict[str, FailureMode] = {m.id: m for m in modes_in_scope}

    items = _collect_items(root, args.mode)
    if args.limit:
        # Keep first N variants per mode (both bad and good).
        keep_variants = {
            mid: sorted({i.variant for i in items if i.mode_id == mid})[: args.limit]
            for mid in mode_by_id
        }
        items = [i for i in items if i.variant in keep_variants[i.mode_id]]

    if not items:
        print("No render.png files found. Did you run `python -m evals.render`?", file=sys.stderr)
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
                f"  {model} {item.mode_id}/{item.variant}/{item.label}: "
                f"API error {e.status_code} — {e.message}",
                file=sys.stderr,
            )
            return None
        except Exception as e:
            print(
                f"  {model} {item.mode_id}/{item.variant}/{item.label}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            traceback.print_exc()
            return None
        return Result(
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
                f"  [{j}/{total}] {model} {item.mode_id}/{item.variant}/{item.label}: "
                f"{ok} (said {'YES' if result.predicted else 'NO'})"
            )

    elapsed = time.time() - started
    print(f"\nDone in {elapsed:.1f}s ({len(results)}/{total} calls succeeded)")

    summary = _summarize(results)
    _print_table(summary)

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
