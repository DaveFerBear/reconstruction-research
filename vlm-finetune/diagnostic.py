"""
Diagnostic: is the trained model actually responding to the corruptions, or
to spurious cues like palette / style / spec identity?

Three sanity checks, all using the baseline-arch model (the winner):

  (1) Per-mode score-gap distribution. For each (clean, corrupted) pair,
      compute s(clean) − s(corrupted). If the model learned regressions,
      these should be positive on average AND vary by mode (the modes the
      model learned best should have the largest mean gap).

  (2) Control: random clean-vs-clean pairs from different specs. If the
      model is responding to corruptions and not to spec identity / style,
      the control mean gap should be near zero, and the "fraction with
      positive gap" should be ~0.50 (no systematic preference).

  (3) Untrained-head baseline. A randomly-initialized MLP (no training)
      should give ~0.50 accuracy on the corruption pairs. If it doesn't,
      UI-CLIP alone is doing the work and the head isn't needed.

Outputs cache/diagnostic_score_gaps.png (overlaid histograms) and
cache/diagnostic_results.json.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from encoder_probe import CACHE_DIR, collect_pairs
from train_baseline_arch import (
    GlobalPairDataset,
    MLPHead,
    bt_loss,
    get_or_cache,
)

OUT_HISTOGRAM = CACHE_DIR / "diagnostic_score_gaps.png"
OUT_RESULTS = CACHE_DIR / "diagnostic_results.json"


def train_full(embeddings, device, epochs=50, lr=1e-3, wd=1e-4, hidden=128, batch_size=64):
    pairs = collect_pairs()
    train_loader = DataLoader(GlobalPairDataset(pairs, embeddings), batch_size=batch_size, shuffle=True)
    model = MLPHead(dim=512, hidden_dim=hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(epochs):
        model.train()
        for clean, corr, _ in train_loader:
            clean, corr = clean.to(device), corr.to(device)
            loss = bt_loss(model(clean), model(corr))
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model


@torch.no_grad()
def score_all(model, paths, embeddings, device):
    model.eval()
    out = {}
    for p in paths:
        x = embeddings[str(p)].unsqueeze(0).to(device)
        out[str(p)] = float(model(x).item())
    return out


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    pairs = collect_pairs()
    paths = sorted({p.clean_path for p in pairs} | {p.corrupted_path for p in pairs})
    embeddings = get_or_cache(paths)
    print(f"Loaded {len(pairs)} pairs ({len(paths)} unique images)")

    # ── Untrained-head baseline ──────────────────────────────────────────────
    print("\n[1/3] Untrained random-init MLP head (no training)...")
    torch.manual_seed(0)
    untrained = MLPHead(dim=512, hidden_dim=128).to(device)
    untrained_scores = score_all(untrained, paths, embeddings, device)
    untrained_correct = sum(
        1 for p in pairs
        if untrained_scores[str(p.clean_path)] > untrained_scores[str(p.corrupted_path)]
    )
    untrained_acc = untrained_correct / len(pairs)
    print(f"  pairwise accuracy: {untrained_acc:.3f}  (chance is 0.500)")

    # ── Trained model ────────────────────────────────────────────────────────
    print("\n[2/3] Training baseline-arch MLP on all 693 corruption pairs...")
    torch.manual_seed(0)
    np.random.seed(0)
    model = train_full(embeddings, device)
    scores = score_all(model, paths, embeddings, device)

    # Per-mode score gaps
    by_mode_gaps: dict[str, list[float]] = defaultdict(list)
    by_mode_correct: dict[str, int] = defaultdict(int)
    by_mode_total: dict[str, int] = defaultdict(int)
    for p in pairs:
        gap = scores[str(p.clean_path)] - scores[str(p.corrupted_path)]
        by_mode_gaps[p.mode].append(gap)
        if gap > 0:
            by_mode_correct[p.mode] += 1
        by_mode_total[p.mode] += 1

    # ── Control: random clean-vs-clean pairs ─────────────────────────────────
    print("\n[3/3] Control: random clean-vs-clean pairs from different specs...")
    rng = random.Random(42)
    clean_paths = sorted({str(p.clean_path) for p in pairs})
    spec_of = {str(p.clean_path): p.spec_id for p in pairs}
    n_control = 200
    control_gaps: list[float] = []
    n_positive = 0
    while len(control_gaps) < n_control:
        a, b = rng.sample(clean_paths, 2)
        if spec_of[a] != spec_of[b]:
            gap = scores[a] - scores[b]
            control_gaps.append(gap)
            if gap > 0:
                n_positive += 1

    # ── Report ───────────────────────────────────────────────────────────────
    ctl = np.array(control_gaps)
    print()
    print("=" * 68)
    print(f"{'group':<28}  {'n':>3}  {'mean gap':>10}  {'sd gap':>8}  {'frac > 0':>10}")
    print("-" * 68)
    for mode in sorted(by_mode_gaps, key=lambda m: -float(np.mean(by_mode_gaps[m]))):
        gaps = np.array(by_mode_gaps[mode])
        acc = by_mode_correct[mode] / by_mode_total[mode]
        print(f"  corruption: {mode:<16}  {len(gaps):>3}  {gaps.mean():>10.3f}  {gaps.std():>8.3f}  {acc:>10.3f}")
    print("-" * 68)
    print(f"  CONTROL (clean vs clean)    {len(ctl):>3}  {ctl.mean():>10.3f}  {ctl.std():>8.3f}  {n_positive / len(ctl):>10.3f}")
    print("-" * 68)
    print(f"  UNTRAINED-HEAD baseline     {len(pairs):>3}  {'—':>10}  {'—':>8}  {untrained_acc:>10.3f}")
    print("=" * 68)
    print()
    print("Interpretation:")
    print("  - Corruption mean gaps should be POSITIVE (clean scored higher than broken).")
    print("  - Per-mode means should DIFFER (modes the model learned best have larger gaps).")
    print("  - Control mean gap should be ~0 and frac > 0 should be ~0.50.")
    print("    A non-zero control mean would mean the model has style/palette preferences,")
    print("    not a learned response to corruptions specifically.")
    print("  - Untrained-head accuracy should be ~0.50. If it's higher, UI-CLIP alone is")
    print("    doing the discrimination and the trained head is redundant.")

    # ── Histogram ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    all_corruption = np.concatenate([np.array(g) for g in by_mode_gaps.values()])
    lo = float(min(ctl.min(), all_corruption.min())) - 0.5
    hi = float(max(ctl.max(), all_corruption.max())) + 0.5
    bins = np.linspace(lo, hi, 50)
    ax.hist(all_corruption, bins=bins, alpha=0.6, label=f"corruption pairs (n={len(all_corruption)})", color="steelblue")
    ax.hist(ctl, bins=bins, alpha=0.6, label=f"control: random clean-vs-clean (n={len(ctl)})", color="orange")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="zero gap")
    ax.set_xlabel("score gap   s(image_a) − s(image_b)")
    ax.set_ylabel("number of pairs")
    ax.set_title(
        "Diagnostic: corruption pairs vs random clean-clean pairs\n"
        "If the model learned the regressions, the orange (control) distribution\n"
        "should sit on top of zero; the blue (corruption) one should sit to the right."
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_HISTOGRAM, dpi=120)
    print(f"\nWrote {OUT_HISTOGRAM}")

    OUT_RESULTS.write_text(json.dumps({
        "untrained_pairwise_acc": untrained_acc,
        "by_mode": {
            m: {
                "n": len(by_mode_gaps[m]),
                "mean_gap": float(np.mean(by_mode_gaps[m])),
                "sd_gap": float(np.std(by_mode_gaps[m])),
                "acc": by_mode_correct[m] / by_mode_total[m],
            }
            for m in by_mode_gaps
        },
        "control": {
            "n": len(ctl),
            "mean_gap": float(ctl.mean()),
            "sd_gap": float(ctl.std()),
            "frac_positive": n_positive / len(ctl),
        },
    }, indent=2))
    print(f"Wrote {OUT_RESULTS}")


if __name__ == "__main__":
    main()
