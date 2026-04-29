"""Render bar charts from `evals/results.json`.

Four charts per source (synthetic / real):
  * `<source>_summary.png`         — HEADLINE: balanced accuracy per mode, by model
  * `<source>_per_mode.png`        — diagnostic: recall@3 + FP rate per mode
  * `<source>_supercategory.png`   — recall@3 / FP rate / bal_acc per layout|visual
  * `<source>_aggregate.png`       — per-model mean bal_acc + mean recall + mean FP

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
    "claude-opus-4-6":     "#1976D2",  # blue
    "claude-sonnet-4-6":   "#F57C00",  # orange
    "gpt-4o":              "#2E7D32",  # green
    "gpt-5":               "#00838F",  # teal
    "ollama/qwen3-vl:4b":  "#7B1FA2",  # purple
}
_FALLBACK_PALETTE = ("#7B1FA2", "#C62828", "#00838F", "#5D4037")

CHANCE = 0.5  # balanced-accuracy expected for a random guesser


def _color(model: str, fallback_idx: int = 0) -> str:
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    return _FALLBACK_PALETTE[fallback_idx % len(_FALLBACK_PALETTE)]


def _v(metric: dict[str, Any] | None, key: str, default: float | None = None) -> float | None:
    if metric is None:
        return default
    val = metric.get(key)
    return default if val is None else float(val)


def _label_bars(ax: plt.Axes, bars, fmt: str = "{:.2f}", *, fontsize: int = 8) -> None:
    """Annotate each bar with its numeric value above the top of the bar."""
    for rect in bars:
        h = rect.get_height()
        if h is None:
            continue
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            h + 0.015,
            fmt.format(h),
            ha="center", va="bottom", fontsize=fontsize, color="#333",
        )


def _max_metric(by_model: dict[str, dict], group: str, key: str) -> float:
    best = 0.0
    for model_data in by_model.values():
        for entry in model_data.get(group, {}).values():
            v = _v(entry, key)
            if v is not None:
                best = max(best, v)
    return best


def _suptitle(fig: plt.Figure, source: str, title: str, subtitle: str = "") -> None:
    """Render a two-line header (title + optional subtitle) at the top of the figure.

    Caller should pair this with `tight_layout(rect=(0, 0, 1, 0.92))` so the
    subplots don't collide with the header. Both lines use va="top" with
    explicit y-coords spaced so they never overlap.
    """
    fig.text(
        0.5, 0.985, f"{source.upper()} — {title}",
        ha="center", va="top", fontsize=14, fontweight="bold",
    )
    if subtitle:
        fig.text(
            0.5, 0.945, subtitle,
            ha="center", va="top", fontsize=10, color="#555", style="italic",
        )


# ---------------------------------------------------------------------------
# 1. Headline: balanced-accuracy per mode
# ---------------------------------------------------------------------------

def _plot_summary(by_model: dict[str, dict], out_path: Path, source: str) -> None:
    models = sorted(by_model.keys())
    if not models:
        return
    modes = list(MODE_DEFINITIONS.keys())
    n_modes, n_models = len(modes), len(models)
    bar_width = 0.8 / max(1, n_models)
    x = np.arange(n_modes)

    fig, ax = plt.subplots(figsize=(max(13, n_modes * 1.2), 6.5))

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * bar_width
        vals: list[float] = []
        for mode in modes:
            ba = _v(by_model[model]["mode"].get(mode), "balanced_accuracy")
            vals.append(ba if ba is not None else 0.0)
        bars = ax.bar(
            x + offset, vals, bar_width,
            label=model, color=_color(model, i), edgecolor="white",
        )
        _label_bars(ax, bars, fontsize=7)

    # Visual chance/perfection guides
    ax.axhline(CHANCE, color="#888", linestyle="--", linewidth=1)
    ax.text(n_modes - 0.4, CHANCE + 0.01, "chance (0.5)", color="#888", fontsize=9, ha="right")
    ax.axhspan(0, CHANCE, color="#FFCDD2", alpha=0.18, zorder=0)  # red tint = below chance

    # Per-mode n_pos / n_neg annotation row under the x-tick labels
    ax.set_xticks(x)
    n_strs = []
    for mode in modes:
        # Use the first model's counts (same across models within a source)
        m = by_model[models[0]]["mode"].get(mode)
        n_pos = m["n_pos"] if m else 0
        n_neg = m["n_neg"] if m else 0
        n_strs.append(f"{mode}\n(n+={n_pos}, n-={n_neg})")
    ax.set_xticklabels(n_strs, rotation=30, ha="right", fontsize=9)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("balanced accuracy  (recall + (1−FP)) / 2")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    _suptitle(
        fig, source,
        "Balanced accuracy per mode",
        "0.5 = chance · 1.0 = perfect · shaded zone = anti-signal",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Diagnostic: recall@3 + FP rate per mode
# ---------------------------------------------------------------------------

def _plot_per_mode(by_model: dict[str, dict], out_path: Path, source: str) -> None:
    models = sorted(by_model.keys())
    if not models:
        return
    modes = list(MODE_DEFINITIONS.keys())
    n_modes, n_models = len(modes), len(models)
    bar_width = 0.8 / max(1, n_models)
    x = np.arange(n_modes)

    fig, (ax_r, ax_f) = plt.subplots(2, 1, figsize=(max(13, n_modes * 1.2), 9), sharex=True)

    fp_max = max(0.4, _max_metric(by_model, "mode", "fp_rate") * 1.15 + 0.05)

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * bar_width
        recalls, fps = [], []
        for mode in modes:
            m = by_model[model]["mode"].get(mode)
            recalls.append(_v(m, "recall_at_3") or 0.0)
            fps.append(_v(m, "fp_rate") or 0.0)
        rb = ax_r.bar(x + offset, recalls, bar_width, label=model,
                      color=_color(model, i), edgecolor="white")
        fb = ax_f.bar(x + offset, fps, bar_width, label=model,
                      color=_color(model, i), edgecolor="white")
        _label_bars(ax_r, rb, fontsize=7)
        _label_bars(ax_f, fb, fontsize=7)

    ax_r.set_ylim(0, 1.05)
    ax_r.set_ylabel("recall@3 (TP / pos)")
    ax_r.set_title("Per-mode recall@3 — did the VLM flag the injected mode in its top 3?")
    ax_r.grid(axis="y", alpha=0.3)
    ax_r.legend(loc="upper right", fontsize=9)

    ax_f.set_ylim(0, fp_max)
    ax_f.set_ylabel("FP rate (FP / neg)")
    ax_f.set_title("Per-mode FP rate — did the VLM flag this mode on a clean original?")
    ax_f.grid(axis="y", alpha=0.3)

    ax_f.set_xticks(x)
    n_strs = []
    for mode in modes:
        m = by_model[models[0]]["mode"].get(mode)
        n_strs.append(f"{mode}\n(n+={m['n_pos'] if m else 0})")
    ax_f.set_xticklabels(n_strs, rotation=30, ha="right", fontsize=9)

    _suptitle(fig, source, "Diagnostic: recall@3 + FP rate by mode", "")
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Supercategory rollup
# ---------------------------------------------------------------------------

def _plot_supercategory(by_model: dict[str, dict], out_path: Path, source: str) -> None:
    models = sorted(by_model.keys())
    if not models:
        return
    cats = ["layout", "visual"]
    n_cats, n_models = len(cats), len(models)
    bar_width = 0.8 / max(1, n_models)
    x = np.arange(n_cats)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_ba, ax_r, ax_f = axes

    fp_max = max(0.6, _max_metric(by_model, "supercategory", "fp_rate") * 1.15 + 0.05)

    panels = [
        (ax_ba, "balanced_accuracy", "balanced accuracy", 1.05, "0.5 = chance"),
        (ax_r,  "recall_at_3",       "recall@3",          1.05, ""),
        (ax_f,  "fp_rate",           "FP rate on originals", fp_max, ""),
    ]
    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * bar_width
        for ax, key, _, _, _ in panels:
            vals = [
                _v(by_model[model]["supercategory"].get(c), key) or 0.0
                for c in cats
            ]
            bars = ax.bar(
                x + offset, vals, bar_width,
                label=model, color=_color(model, i), edgecolor="white",
            )
            _label_bars(ax, bars, fontsize=8)

    for ax, key, ylabel, ymax, sub in panels:
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
        title = f"{ylabel}\n{sub}" if sub else ylabel
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if key == "balanced_accuracy":
            ax.axhline(CHANCE, color="#888", linestyle="--", linewidth=1)
            ax.axhspan(0, CHANCE, color="#FFCDD2", alpha=0.15, zorder=0)
    axes[0].legend(loc="upper right", fontsize=9)

    _suptitle(fig, source, "Supercategory comparison (LAYOUT vs VISUAL)", "")
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Per-model aggregate
# ---------------------------------------------------------------------------

def _plot_aggregate(by_model: dict[str, dict], out_path: Path, source: str) -> None:
    models = sorted(by_model.keys())
    if not models:
        return
    bas     = [_v(by_model[m]["aggregate"], "mean_balanced_accuracy") or 0.0 for m in models]
    recalls = [_v(by_model[m]["aggregate"], "mean_recall_at_3") or 0.0 for m in models]
    fps     = [_v(by_model[m]["aggregate"], "mean_fp_rate") or 0.0 for m in models]
    n_judgments = [by_model[m]["aggregate"].get("n_judgments", 0) for m in models]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5.5))
    x = np.arange(len(models))
    width = 0.27

    bb = ax.bar(x - width, bas,     width, label="mean balanced accuracy", color="#37474F", edgecolor="white")
    rb = ax.bar(x,         recalls, width, label="mean recall@3",          color="#1976D2", edgecolor="white")
    fb = ax.bar(x + width, fps,     width, label="mean FP rate",            color="#C62828", edgecolor="white")
    _label_bars(ax, bb, fontsize=9)
    _label_bars(ax, rb, fontsize=9)
    _label_bars(ax, fb, fontsize=9)

    ax.axhline(CHANCE, color="#888", linestyle="--", linewidth=1)
    ax.text(len(models) - 0.5, CHANCE + 0.01, "chance (bal_acc=0.5)", color="#888", fontsize=9, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(n={nj})" for m, nj in zip(models, n_judgments)], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    _suptitle(
        fig, source,
        "Per-model aggregate (averaged across modes)",
        "Balanced accuracy is the headline; recall + FP rate decompose it.",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def make_charts(summary: dict[str, dict], out_dir: Path) -> list[Path]:
    """Render charts for every source in `summary`. Returns paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source, by_model in summary.items():
        if not by_model:
            continue
        files = {
            "summary":       out_dir / f"{source}_summary.png",
            "per_mode":      out_dir / f"{source}_per_mode.png",
            "supercategory": out_dir / f"{source}_supercategory.png",
            "aggregate":     out_dir / f"{source}_aggregate.png",
        }
        _plot_summary(by_model, files["summary"], source)
        _plot_per_mode(by_model, files["per_mode"], source)
        _plot_supercategory(by_model, files["supercategory"], source)
        _plot_aggregate(by_model, files["aggregate"], source)
        written.extend(files.values())
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
