"""Run the open-ended VLM-as-judge eval across failure modes and models.

For each rendered design, ask each VLM to list its top-3 design issues
(free-text, no taxonomy primer). Then classify every emitted issue against
our 11-mode taxonomy via a Haiku classifier. Score each (model, mode) by
recall@3 (did the VLM flag the injected mode in its top-3?) and FP rate
(did it flag the mode on a known-clean original?). Roll modes up into two
supercategories — `layout` and `visual` — for a coarser-grained view.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evals.classify import IssueCategory, classify_batch
from evals.common import MODE_DEFINITIONS, SUPERCATEGORIES
from evals.judge import DEFAULT_MODELS, IssueSet, judge

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DATA_DIR = Path(__file__).parent / "data"
REAL_DATA_DIR = Path(__file__).parent / "data_real"
SOURCE_DIR = REPO_ROOT / "datasets" / "specs"
RESULTS_PATH = Path(__file__).parent / "results.json"


@dataclass
class Render:
    source: str             # "synthetic" | "real"
    spec_id: str            # synthetic: "<mode>/<NN>"; real: spec dir name
    render_path: Path
    is_bad: bool            # True iff a corruption was injected
    injected_mode: str | None  # mode_id if bad, None otherwise

    @property
    def render_id(self) -> str:
        # Must include injected_mode so different per-mode corruptions of the
        # same spec don't collapse to the same key. Otherwise --incremental
        # treats "the OVERFLOW corruption of spec X" and "the OVERLAP
        # corruption of spec X" as the same render and silently skips.
        mode = self.injected_mode or "original"
        return f"{self.source}::{self.spec_id}::{mode}"


@dataclass
class Judgment:
    render_id: str
    source: str
    spec_id: str
    is_bad: bool
    injected_mode: str | None
    render_path: str
    model: str
    issues: list[str]
    raw: str
    parse_error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    classifications: list[dict[str, Any]] = field(default_factory=list)


def _collect_synthetic(root: Path, only_mode: str | None) -> list[Render]:
    out: list[Render] = []
    if not root.exists():
        return out
    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir():
            continue
        if only_mode and mode_dir.name != only_mode:
            continue
        for variant_dir in sorted(mode_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            bad = variant_dir / "bad" / "render.png"
            good = variant_dir / "good" / "render.png"
            spec_id = f"{mode_dir.name}/{variant_dir.name}"
            if bad.exists():
                out.append(Render(
                    source="synthetic", spec_id=spec_id,
                    render_path=bad, is_bad=True, injected_mode=mode_dir.name,
                ))
            if good.exists():
                out.append(Render(
                    source="synthetic", spec_id=spec_id,
                    render_path=good, is_bad=False, injected_mode=None,
                ))
    return out


def _collect_real(real_root: Path, source_root: Path, only_mode: str | None) -> list[Render]:
    out: list[Render] = []
    if not real_root.exists():
        return out
    seen_originals: set[str] = set()
    for spec_dir in sorted(real_root.iterdir()):
        if not spec_dir.is_dir():
            continue
        original = source_root / spec_dir.name / "render.png"
        for mode_dir in sorted(spec_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            if only_mode and mode_dir.name != only_mode:
                continue
            corrupted = mode_dir / "render.png"
            if corrupted.exists():
                out.append(Render(
                    source="real", spec_id=spec_dir.name,
                    render_path=corrupted, is_bad=True, injected_mode=mode_dir.name,
                ))
        if original.exists() and spec_dir.name not in seen_originals:
            seen_originals.add(spec_dir.name)
            out.append(Render(
                source="real", spec_id=spec_dir.name,
                render_path=original, is_bad=False, injected_mode=None,
            ))
    return out


def _apply_limit(renders: list[Render], limit: int) -> list[Render]:
    """Keep the first `limit` distinct spec_ids per (source, injected_mode), so
    `--limit 5` becomes a small but still-balanced smoke test."""
    keep: dict[tuple[str, str | None], list[str]] = {}
    for r in renders:
        key = (r.source, r.injected_mode)
        keep.setdefault(key, [])
        if r.spec_id not in keep[key]:
            keep[key].append(r.spec_id)
    for key in keep:
        keep[key] = keep[key][:limit]
    return [r for r in renders if r.spec_id in keep[(r.source, r.injected_mode)]]


def _summarize(
    judgments: list[Judgment],
    classifications: dict[str, IssueCategory],
) -> dict:
    """Build per-(source, model, mode) and per-(source, model, supercategory)
    recall@3 + FP rate. Aggregates per (source, model)."""

    def cats_for(j: Judgment) -> list[IssueCategory]:
        return [classifications.get(t, IssueCategory(None, None)) for t in j.issues]

    sources = sorted({j.source for j in judgments})
    models = sorted({j.model for j in judgments})
    summary: dict[str, dict[str, dict[str, Any]]] = {}

    for source in sources:
        summary[source] = {}
        for model in models:
            sm = [j for j in judgments if j.source == source and j.model == model]
            mode_metrics: dict[str, dict[str, Any]] = {}
            for mode_id in MODE_DEFINITIONS:
                positives = [j for j in sm if j.is_bad and j.injected_mode == mode_id]
                negatives = [j for j in sm if not j.is_bad]
                tp = sum(1 for j in positives if any(c.mode_id == mode_id for c in cats_for(j)))
                fp = sum(1 for j in negatives if any(c.mode_id == mode_id for c in cats_for(j)))
                recall = tp / len(positives) if positives else None
                fp_rate = fp / len(negatives) if negatives else None
                ba = ((recall + (1 - fp_rate)) / 2) if (recall is not None and fp_rate is not None) else None
                mode_metrics[mode_id] = {
                    "recall_at_3":       round(recall, 3) if recall is not None else None,
                    "fp_rate":           round(fp_rate, 3) if fp_rate is not None else None,
                    "balanced_accuracy": round(ba, 3) if ba is not None else None,
                    "n_pos": len(positives),
                    "n_neg": len(negatives),
                    "tp": tp, "fp": fp,
                }

            super_metrics: dict[str, dict[str, Any]] = {}
            for super_name in ("layout", "visual"):
                modes_in = {m for m, s in SUPERCATEGORIES.items() if s == super_name}
                positives = [j for j in sm if j.is_bad and j.injected_mode in modes_in]
                negatives = [j for j in sm if not j.is_bad]
                tp = sum(
                    1 for j in positives
                    if any(c.supercategory == super_name for c in cats_for(j))
                )
                fp = sum(
                    1 for j in negatives
                    if any(c.supercategory == super_name for c in cats_for(j))
                )
                recall = tp / len(positives) if positives else None
                fp_rate = fp / len(negatives) if negatives else None
                ba = ((recall + (1 - fp_rate)) / 2) if (recall is not None and fp_rate is not None) else None
                super_metrics[super_name] = {
                    "recall_at_3":       round(recall, 3) if recall is not None else None,
                    "fp_rate":           round(fp_rate, 3) if fp_rate is not None else None,
                    "balanced_accuracy": round(ba, 3) if ba is not None else None,
                    "n_pos": len(positives),
                    "n_neg": len(negatives),
                    "tp": tp, "fp": fp,
                }

            recalls = [m["recall_at_3"] for m in mode_metrics.values() if m["recall_at_3"] is not None]
            fps     = [m["fp_rate"]     for m in mode_metrics.values() if m["fp_rate"]     is not None]
            bas     = [m["balanced_accuracy"] for m in mode_metrics.values() if m["balanced_accuracy"] is not None]
            aggregate = {
                "mean_recall_at_3":        round(sum(recalls) / len(recalls), 3) if recalls else None,
                "mean_fp_rate":            round(sum(fps) / len(fps), 3) if fps else None,
                "mean_balanced_accuracy":  round(sum(bas) / len(bas), 3) if bas else None,
                "n_judgments": len(sm),
            }
            summary[source][model] = {
                "mode": mode_metrics,
                "supercategory": super_metrics,
                "aggregate": aggregate,
            }
    return summary


def _print_tables(summary: dict, source: str) -> None:
    s = summary.get(source)
    if not s:
        print(f"(no data for source={source})")
        return
    models = sorted(s.keys())
    print(f"\n========== {source.upper()} ==========\n")
    print("Cell format: bal_acc / recall@3 / FP rate  (n_pos/n_neg)")
    print("  bal_acc 0.5 = chance | 1.0 = perfect | <0.5 = anti-signal\n")

    # Per-mode table
    print("Per mode")
    header = f"{'mode':<24}" + "".join(f"{m:<28}" for m in models)
    print(header)
    print("-" * len(header))
    for mode_id in MODE_DEFINITIONS:
        row = f"{mode_id:<24}"
        for model in models:
            m = s[model]["mode"][mode_id]
            ba = m.get("balanced_accuracy")
            r = m["recall_at_3"]
            f = m["fp_rate"]
            ba_str = f"{ba:.2f}" if ba is not None else "  - "
            r_str = f"{r:.2f}" if r is not None else "  - "
            f_str = f"{f:.2f}" if f is not None else "  - "
            cell = f"{ba_str}/{r_str}/{f_str} ({m['n_pos']}/{m['n_neg']})"
            row += f"{cell:<28}"
        print(row)

    # Per-supercategory table
    print("\nPer supercategory")
    print(header)
    print("-" * len(header))
    for super_name in ("layout", "visual"):
        row = f"{super_name:<24}"
        for model in models:
            m = s[model]["supercategory"][super_name]
            ba = m.get("balanced_accuracy")
            r = m["recall_at_3"]
            f = m["fp_rate"]
            ba_str = f"{ba:.2f}" if ba is not None else "  - "
            r_str = f"{r:.2f}" if r is not None else "  - "
            f_str = f"{f:.2f}" if f is not None else "  - "
            cell = f"{ba_str}/{r_str}/{f_str} ({m['n_pos']}/{m['n_neg']})"
            row += f"{cell:<28}"
        print(row)

    # Aggregate per model
    print("\nAggregate per model")
    print(f"{'model':<24}{'mean bal_acc':<16}{'mean recall@3':<18}{'mean FP rate':<18}{'#judgments':<12}")
    print("-" * 90)
    for model in models:
        agg = s[model]["aggregate"]
        ba = agg.get("mean_balanced_accuracy")
        r  = agg["mean_recall_at_3"]
        f  = agg["mean_fp_rate"]
        ba_s = f"{ba:.3f}" if ba is not None else "  - "
        r_s  = f"{r:.3f}"  if r  is not None else "  - "
        f_s  = f"{f:.3f}"  if f  is not None else "  - "
        print(f"{model:<24}{ba_s:<16}{r_s:<18}{f_s:<18}{agg['n_judgments']:<12}")
    print()


def _run_judging(
    renders: list[Render],
    models: tuple[str, ...],
    concurrency: int,
    *,
    verbose: bool = False,
) -> list[Judgment]:
    jobs: list[tuple[Render, str]] = [(r, m) for r in renders for m in models]
    total = len(jobs)
    print(f"Judging {len(renders)} renders × {len(models)} models = {total} calls (concurrency={concurrency})")

    def _go(job: tuple[Render, str]) -> Judgment | None:
        render, model = job
        try:
            result: IssueSet = judge(render.render_path, model)
        except Exception as e:
            tqdm.write(
                f"judge failed: {model} {render.render_id}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            traceback.print_exc()
            return None
        return Judgment(
            render_id=render.render_id,
            source=render.source,
            spec_id=render.spec_id,
            is_bad=render.is_bad,
            injected_mode=render.injected_mode,
            render_path=str(render.render_path),
            model=model,
            issues=result.issues,
            raw=result.raw,
            parse_error=result.parse_error,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    judgments: list[Judgment] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(_go, j): j for j in jobs}
        bar = tqdm(as_completed(futures), total=total, desc="judging", unit="img")
        for future in bar:
            r, model = futures[future]
            j = future.result()
            if j is None:
                continue
            judgments.append(j)
            if verbose:
                summary = "; ".join(s[:50] for s in j.issues) or "(no issues)"
                if j.parse_error:
                    summary = f"PARSE ERR: {j.parse_error}"
                tqdm.write(f"  {model} {r.render_id}: {summary[:120]}")
    elapsed = time.time() - started
    print(f"Judging done in {elapsed:.1f}s ({len(judgments)}/{total} succeeded)")
    return judgments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the open-ended VLM-blindness eval")
    parser.add_argument(
        "--source",
        choices=["synthetic", "real", "both"],
        default="both",
        help="Which dataset(s) to evaluate (default: both)",
    )
    parser.add_argument(
        "--mode",
        help="Restrict to one failure-mode id (filters injected_mode for bads)",
    )
    parser.add_argument(
        "--model",
        action="append",
        help=f"Model (repeatable). Default: {', '.join(DEFAULT_MODELS)}",
    )
    parser.add_argument("--limit", type=int, help="Cap variants/specs per mode for smoke tests")
    parser.add_argument(
        "--synthetic-root", default=str(SYNTHETIC_DATA_DIR),
        help=f"Synthetic data root (default: {SYNTHETIC_DATA_DIR})",
    )
    parser.add_argument(
        "--real-root", default=str(REAL_DATA_DIR),
        help=f"Real corruption data root (default: {REAL_DATA_DIR})",
    )
    parser.add_argument(
        "--source-root", default=str(SOURCE_DIR),
        help=f"Original specs root for real-source goods (default: {SOURCE_DIR})",
    )
    parser.add_argument("--out", default=str(RESULTS_PATH), help=f"Results JSON path (default: {RESULTS_PATH})")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent API calls (default: 8)")
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        help="Don't classify; uncached issues map to (None, None). Useful for table re-renders.",
    )
    parser.add_argument(
        "--reuse-judgments",
        action="store_true",
        help="Read existing results.json instead of re-judging. Reclassifies + recomputes summary.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Reuse any cached judgments from results.json; judge only the "
            "(render, model) pairs that aren't already there. Use this to add "
            "a new model to a previous run without redoing the old ones."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print each judgment's parsed issues alongside the progress bar.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip the chart-rendering step at the end.",
    )
    args = parser.parse_args()

    models = tuple(args.model) if args.model else DEFAULT_MODELS

    # Fail fast on missing ollama dep so we don't burn through 200 judge calls
    # discovering it the hard way.
    if any(m.startswith("ollama/") for m in models):
        try:
            import ollama  # noqa: F401, PLC0415
        except ImportError:
            print(
                "Models include ollama/* but the `ollama` package isn't installed.\n"
                "Run `pip install ollama` (or `pip install -r requirements.txt`),\n"
                "and make sure the Ollama daemon is running (`ollama serve`).",
                file=sys.stderr,
            )
            sys.exit(2)

    renders: list[Render] = []
    if args.source in ("synthetic", "both"):
        renders.extend(_collect_synthetic(Path(args.synthetic_root), args.mode))
    if args.source in ("real", "both"):
        renders.extend(_collect_real(Path(args.real_root), Path(args.source_root), args.mode))
    if args.limit:
        renders = _apply_limit(renders, args.limit)
    if not renders:
        print("No renders found. Did generate + render run?", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)

    judgments: list[Judgment]
    cached_judgments: list[Judgment] = []
    if (args.reuse_judgments or args.incremental) and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        raw_judgments = existing.get("judgments", [])
        valid_render_ids = {r.render_id for r in renders}
        valid_models = set(models)
        cached_judgments = [
            Judgment(**{k: v for k, v in j.items() if k in Judgment.__dataclass_fields__})
            for j in raw_judgments
            if j.get("render_id") in valid_render_ids and j.get("model") in valid_models
        ]

    if args.reuse_judgments:
        judgments = cached_judgments
        print(f"Reusing {len(judgments)} judgments from {out_path}")
    elif args.incremental:
        seen = {(j.render_id, j.model) for j in cached_judgments}
        missing_renders_models: dict[str, set[str]] = {}
        for r in renders:
            for m in models:
                if (r.render_id, m) not in seen:
                    missing_renders_models.setdefault(r.render_id, set()).add(m)
        missing_jobs = [
            (r, m)
            for r in renders
            for m in models
            if (r.render_id, m) not in seen
        ]
        print(
            f"Incremental: {len(cached_judgments)} cached, "
            f"{len(missing_jobs)} new (render, model) pairs to judge."
        )
        if missing_jobs:
            # Reuse _run_judging by reconstructing a (render, model) cross-product
            # that contains only the pairs we still need. Dedupe by render_id
            # (Render isn't hashable — it has no @dataclass(frozen=True)).
            renders_by_id: dict[str, Render] = {r.render_id: r for r, _ in missing_jobs}
            renders_needed = list(renders_by_id.values())
            models_per_render: dict[str, set[str]] = {rid: set() for rid in renders_by_id}
            for r, m in missing_jobs:
                models_per_render[r.render_id].add(m)
            # Constraint: _run_judging takes a tuple of models and runs the full cross-product.
            # If different renders need different model subsets, do it per model.
            new_judgments: list[Judgment] = []
            for m in models:
                renders_for_model = [
                    r for r in renders_needed if m in models_per_render[r.render_id]
                ]
                if not renders_for_model:
                    continue
                new_judgments.extend(_run_judging(
                    renders_for_model, (m,),
                    concurrency=args.concurrency, verbose=args.verbose,
                ))
            judgments = cached_judgments + new_judgments
        else:
            judgments = cached_judgments
    else:
        judgments = _run_judging(
            renders, models, concurrency=args.concurrency, verbose=args.verbose
        )

    # Classify every unique issue string (cached by SHA256 across runs)
    all_issues = [text for j in judgments for text in j.issues]
    if args.skip_classifier:
        # Map only via cache (no API calls)
        from evals.classify import _Cache, CACHE_PATH  # noqa: PLC0415
        cache = _Cache(CACHE_PATH)
        cats: list[IssueCategory] = []
        for t in all_issues:
            hit = cache.get(t)
            cats.append(
                IssueCategory(mode_id=hit["mode"], supercategory=hit["supercategory"])
                if hit else IssueCategory(None, None)
            )
    else:
        cats = classify_batch(all_issues, concurrency=args.concurrency)

    by_text: dict[str, IssueCategory] = {}
    for text, cat in zip(all_issues, cats):
        by_text.setdefault(text, cat)

    # Attach classifications to each judgment for the JSON output
    for j in judgments:
        j.classifications = [
            {"text": t, "mode": by_text[t].mode_id, "supercategory": by_text[t].supercategory}
            for t in j.issues
        ]

    summary = _summarize(judgments, by_text)
    for source in sorted(summary):
        _print_tables(summary, source)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "judgments": [asdict(j) for j in judgments],
        "classifications": {
            t: {"text": t, "mode": c.mode_id, "supercategory": c.supercategory}
            for t, c in by_text.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    if not args.no_charts:
        from evals.make_charts import make_charts  # noqa: PLC0415

        chart_dir = out_path.parent / "charts"
        make_charts(payload["summary"], chart_dir)


if __name__ == "__main__":
    main()
