"""Render bar charts from `evals/results.json`.

Three charts per source (synthetic / real):
  * `<source>_per_mode.png`        — recall@3 + FP rate per mode, grouped by model
  * `<source>_supercategory.png`   — recall@3 + FP rate per supercategory, by model
  * `<source>_aggregate.png`       — per-model mean recall@3 + mean FP rate

Auto-invoked by `run_eval.py` after a run; also runnable standalone:

    python -m evals.make_charts
    python -m evals.make_charts --results evals/results.json --out evals/charts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evals.common import MODE_DEFINITIONS, SUPERCATEGORIES

DEFAULT_RESULTS = Path(__file__).parent / "results.json"
DEFAULT_OUT = Path(__file__).parent / "charts"

# Stable per-model colors so charts are visually consistent across runs.
MODEL_COLORS: dict[str, str] = {
    "claude-opus-4-6":   "#1976D2",  # blue
    "claude-sonnet-4-6": "#F57C00",  # orange
    "gpt-4o":            "#2E7D32",  # green
}
_FALLBACK_PALETTE = ("#7B1FA2", "#C62828", "#00838F", "#5D4037")


def _color(model: str, fallback_idx: int = 0) -> str:
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    return _FALLBACK_PALETTE[fallback_idx % len(_FALLBACK_PALETTE)]


def _value(metric: dict[str, Any] | None, key: str) -> float | None:
    if metric is None:
        return None
    v = metric.get(key)
    return None if v is None else float(v)


def _plot_per_mode(by_model: dict[str, dict], out_path: Path, source: str) -> None:
    models = sorted(by_model.keys())
    if not models:
        return
    modes = list(MODE_DEFINITIONS.keys())
    n_modes = len(modes)
    n_models = len(models)
    bar_width = 0.8 / max(1, n_models)
    x = np.arange(n_modes)

    fig, (ax_r, ax_f) = plt.subplots(2, 1, figsize=(max(12, n_modes * 1.1), 9), sharex=True)

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * bar_width
        recalls: list[float] = []
        fps: list[float] = []
        n_pos: list[int] = []
        for mode in modes:
            m = by_model[model]["mode"].get(mode)
            r = _value(m, "recall_at_3")
            f = _value(m, "fp_rate")
            recalls.append(r if r is not None else 0.0)
            fps.append(f if f is not None else 0.0)
            n_pos.append(int(m["n_pos"]) if m else 0)
        ax_r.bar(x + offset, recalls, bar_width, label=model, color=_color(model, i), edgecolor="white")
        ax_f.bar(x + offset, fps, bar_width, label=model, color=_color(model, i), edgecolor="white")

    ax_r.set_ylim(0, 1)
    ax_r.set_ylabel("recall@3 (higher = better)")
    ax_r.set_title(f"{source.upper()} — per-mode recall@3")
    ax_r.grid(axis="y", alpha=0.3)
    ax_r.legend(loc="upper right", fontsize=9)
    ax_r.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)

    ax_f.set_ylim(0, max(0.5, max(_max_metric(by_model, "mode", "fp_rate"), 0.0) * 1.15 + 0.05))
    ax_f.set_ylabel("FP rate on originals (lower = better)")
    ax_f.set_title("per-mode false-positive rate")
    ax_f.grid(axis="y", alpha=0.3)
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(modes, rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_supercategory(by_model: dict[str, dict], out_path: Path, source: str) -> None:
    models = sorted(by_model.keys())
    if not models:
        return
    cats = ["layout", "visual"]
    n_cats = len(cats)
    n_models = len(models)
    bar_width = 0.8 / max(1, n_models)
    x = np.arange(n_cats)

    fig, (ax_r, ax_f) = plt.subplots(1, 2, figsize=(11, 5))

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * bar_width
        recalls = []
        fps = []
        for cat in cats:
            m = by_model[model]["supercategory"].get(cat)
            recalls.append(_value(m, "recall_at_3") or 0.0)
            fps.append(_value(m, "fp_rate") or 0.0)
        ax_r.bar(x + offset, recalls, bar_width, label=model, color=_color(model, i), edgecolor="white")
        ax_f.bar(x + offset, fps, bar_width, label=model, color=_color(model, i), edgecolor="white")

    for ax, ylabel, title, ymax in (
        (ax_r, "recall@3 (higher = better)", "supercategory recall@3", 1.0),
        (ax_f, "FP rate (lower = better)",   "supercategory FP rate",
         max(0.5, max(_max_metric(by_model, "supercategory", "fp_rate"), 0.0) * 1.15 + 0.05)),
    ):
        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"{source.upper()} — supercategory comparison", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_aggregate(by_model: dict[str, dict], out_path: Path, source: str) -> None:
    models = sorted(by_model.keys())
    if not models:
        return
    means_recall = [
        by_model[m]["aggregate"].get("mean_recall_at_3") or 0.0 for m in models
    ]
    means_fp = [
        by_model[m]["aggregate"].get("mean_fp_rate") or 0.0 for m in models
    ]

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.6), 5))
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, means_recall, width, label="mean recall@3", color="#1976D2", edgecolor="white")
    ax.bar(x + width / 2, means_fp,     width, label="mean FP rate",  color="#C62828", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title(f"{source.upper()} — per-model aggregate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for i, (r, f) in enumerate(zip(means_recall, means_fp)):
        ax.text(i - width / 2, r + 0.015, f"{r:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, f + 0.015, f"{f:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _max_metric(by_model: dict[str, dict], group: str, key: str) -> float:
    """Helper to find the largest non-None value across all (model, mode/super) of a metric."""
    best = 0.0
    for model_data in by_model.values():
        for entry in model_data.get(group, {}).values():
            if entry and entry.get(key) is not None:
                best = max(best, float(entry[key]))
    return best


def make_charts(summary: dict[str, dict], out_dir: Path) -> list[Path]:
    """Render charts for every source in `summary`. Returns paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source, by_model in summary.items():
        if not by_model:
            continue
        per_mode_path = out_dir / f"{source}_per_mode.png"
        super_path    = out_dir / f"{source}_supercategory.png"
        agg_path      = out_dir / f"{source}_aggregate.png"
        _plot_per_mode(by_model, per_mode_path, source)
        _plot_supercategory(by_model, super_path, source)
        _plot_aggregate(by_model, agg_path, source)
        written.extend([per_mode_path, super_path, agg_path])
    print(f"Wrote {len(written)} chart(s) to {out_dir}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Render charts from results.json")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help=f"Results JSON path (default: {DEFAULT_RESULTS})")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"Output directory (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"No results found at {results_path}; run `python -m evals.run_eval` first.")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    make_charts(summary, Path(args.out))


if __name__ == "__main__":
    main()
