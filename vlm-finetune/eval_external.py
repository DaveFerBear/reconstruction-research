"""
Phase 2: cross-distribution eval of the BT ranker.

Trains 3 seeded models on all 693 corruption pairs (no held-out) and evaluates
each on two external sets:
  - Synthetic: evals/data/<mode>/NN/{good,bad}/render.png  (~45 hand-built pairs)
  - Wild:      vlm-finetune/dataset/pairs_hard.jsonl       (554 Gemini-scored agent edits, |Δ|≥10)

Reports overall pairwise accuracy + per-label breakdowns (mode for synthetic,
edit_type_name for wild). Wild ground truth is noisy (single Gemini call per
pair), so even a perfect ranker would not reach 1.0 there.

Usage:
  pai3 && python vlm-finetune/eval_external.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from encoder_probe import CACHE_DIR, REPO_ROOT, collect_pairs
from train import (
    EMBED_DIM,
    AttentionPoolHead,
    PairDataset,
    bt_loss,
    embed_native_crops,
    get_or_cache_embeddings,
    pad_collate,
)

WILD_DATASET_DIR = Path(__file__).parent / "dataset"
WILD_PAIRS_FILE = WILD_DATASET_DIR / "pairs_hard.jsonl"
SYNTHETIC_DATA_DIR = REPO_ROOT / "evals" / "data"
WILD_CACHE = CACHE_DIR / "uiclip_native_crops_wild.pt"
SYNTHETIC_CACHE = CACHE_DIR / "uiclip_native_crops_synthetic.pt"


def load_wild_pairs() -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    with open(WILD_PAIRS_FILE) as f:
        for line in f:
            r = json.loads(line)
            aes = WILD_DATASET_DIR / r["aesthetic_path"]
            neg = WILD_DATASET_DIR / r["non_aesthetic_path"]
            if not aes.exists() or not neg.exists():
                continue
            label = r.get("edit_type_name") or "unknown"
            pairs.append((str(aes), str(neg), label))
    return pairs


def load_synthetic_pairs() -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    if not SYNTHETIC_DATA_DIR.exists():
        return pairs
    for mode_dir in sorted(SYNTHETIC_DATA_DIR.iterdir()):
        if not mode_dir.is_dir():
            continue
        for variant_dir in sorted(mode_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            good = variant_dir / "good" / "render.png"
            bad = variant_dir / "bad" / "render.png"
            if good.exists() and bad.exists():
                pairs.append((str(good), str(bad), mode_dir.name))
    return pairs


def get_or_cache_external_embeddings(paths: list[str], cache_path: Path) -> dict[str, torch.Tensor]:
    cached: dict[str, torch.Tensor] = {}
    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
    missing = [Path(p) for p in paths if p not in cached]
    if missing:
        print(f"  embedding {len(missing)} new images for cache {cache_path.name}")
        new = embed_native_crops(missing)
        cached.update(new)
        torch.save(cached, cache_path)
    return cached


@torch.no_grad()
def score_pair(model: AttentionPoolHead, embeddings: dict, path_a: str, path_b: str, device: str) -> tuple[float, float]:
    a = embeddings[path_a].unsqueeze(0).to(device)
    b = embeddings[path_b].unsqueeze(0).to(device)
    mask_a = torch.ones(1, a.shape[1], dtype=torch.bool, device=device)
    mask_b = torch.ones(1, b.shape[1], dtype=torch.bool, device=device)
    s_a = model(a, mask_a).item()
    s_b = model(b, mask_b).item()
    return float(s_a), float(s_b)


def evaluate_external(
    model: AttentionPoolHead,
    pairs: list[tuple[str, str, str]],
    embeddings: dict,
    device: str,
) -> dict:
    model.eval()
    correct = 0
    by_label: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    diffs: list[tuple[float, int]] = []
    for path_aes, path_neg, label in pairs:
        s_aes, s_neg = score_pair(model, embeddings, path_aes, path_neg, device)
        is_correct = int(s_aes > s_neg)
        correct += is_correct
        by_label[label][0] += is_correct
        by_label[label][1] += 1
        diffs.append((s_aes - s_neg, is_correct))
    return {
        "overall": correct / len(pairs) if pairs else 0.0,
        "n": len(pairs),
        "by_label": {k: {"acc": v[0] / v[1], "correct": v[0], "n": v[1]} for k, v in by_label.items()},
        "mean_score_diff": float(np.mean([d[0] for d in diffs])) if diffs else 0.0,
    }


def train_full_model(
    embeddings: dict,
    epochs: int,
    batch_size: int,
    lr: float,
    wd: float,
    device: str,
) -> AttentionPoolHead:
    pairs = collect_pairs()
    train_ds = PairDataset(pairs, embeddings)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    model = AttentionPoolHead(dim=EMBED_DIM, n_heads=4, mlp_hidden=256, dropout=0.1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_b = 0
        for clean_x, clean_m, corr_x, corr_m, _ in train_loader:
            clean_x, clean_m = clean_x.to(device), clean_m.to(device)
            corr_x, corr_m = corr_x.to(device), corr_m.to(device)
            s_clean = model(clean_x, clean_m)
            s_corr = model(corr_x, corr_m)
            loss = bt_loss(s_clean, s_corr)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_b += 1
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  epoch {epoch:3d}  loss={total_loss / n_b:.3f}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    train_pairs = collect_pairs()
    train_paths = sorted({p.clean_path for p in train_pairs} | {p.corrupted_path for p in train_pairs})
    train_embs = get_or_cache_embeddings(train_paths)

    wild_pairs = load_wild_pairs()
    synth_pairs = load_synthetic_pairs()
    print(f"Wild:      {len(wild_pairs)} pairs from {WILD_PAIRS_FILE}")
    print(f"Synthetic: {len(synth_pairs)} pairs from {SYNTHETIC_DATA_DIR}")

    wild_paths = sorted({p[0] for p in wild_pairs} | {p[1] for p in wild_pairs})
    synth_paths = sorted({p[0] for p in synth_pairs} | {p[1] for p in synth_pairs})
    wild_embs = get_or_cache_external_embeddings(wild_paths, WILD_CACHE)
    synth_embs = get_or_cache_external_embeddings(synth_paths, SYNTHETIC_CACHE)

    wild_results: list[dict] = []
    synth_results: list[dict] = []
    for seed in range(args.seeds):
        print(f"\n=== Training final model on all {len(train_pairs)} corruption pairs, seed={seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = train_full_model(train_embs, args.epochs, args.batch_size, args.lr, args.wd, device)
        wild_res = evaluate_external(model, wild_pairs, wild_embs, device)
        synth_res = evaluate_external(model, synth_pairs, synth_embs, device)
        wild_res["seed"] = seed
        synth_res["seed"] = seed
        print(f"  Synthetic:  overall={synth_res['overall']:.3f}  n={synth_res['n']}")
        print(f"  Wild:       overall={wild_res['overall']:.3f}  n={wild_res['n']}")
        wild_results.append(wild_res)
        synth_results.append(synth_res)

    print("\n" + "=" * 60)
    wild_acc = np.array([r["overall"] for r in wild_results])
    synth_acc = np.array([r["overall"] for r in synth_results])

    def fmt_ci(arr: np.ndarray) -> str:
        n = len(arr)
        if n <= 1:
            return f"{arr[0]:.3f}"
        mean = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(n)
        return f"{mean:.3f} ± {arr.std(ddof=1):.3f} (95% CI [{mean - 1.96 * se:.3f}, {mean + 1.96 * se:.3f}])"

    print(f"Synthetic ({synth_results[0]['n']} pairs):  {fmt_ci(synth_acc)}")
    print(f"Wild      ({wild_results[0]['n']} pairs): {fmt_ci(wild_acc)}")

    def agg_by_label(results: list[dict]) -> dict:
        all_labels: set[str] = set()
        for r in results:
            all_labels.update(r["by_label"].keys())
        out: dict[str, dict] = {}
        for label in sorted(all_labels):
            accs = [r["by_label"][label]["acc"] for r in results if label in r["by_label"]]
            ns = [r["by_label"][label]["n"] for r in results if label in r["by_label"]]
            out[label] = {
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                "n": int(np.mean(ns)),
            }
        return out

    print("\nSynthetic — pairwise accuracy by mode (held-out from training):")
    synth_agg = agg_by_label(synth_results)
    for label, stats in sorted(synth_agg.items(), key=lambda kv: -kv[1]["acc_mean"]):
        print(f"  {label:25s}  acc={stats['acc_mean']:.3f} ± {stats['acc_std']:.3f}  n={stats['n']}")

    print("\nWild — pairwise accuracy by edit_type_name (Gemini-noisy ground truth):")
    wild_agg = agg_by_label(wild_results)
    for label, stats in sorted(wild_agg.items(), key=lambda kv: -kv[1]["acc_mean"]):
        print(f"  {label:30s}  acc={stats['acc_mean']:.3f} ± {stats['acc_std']:.3f}  n={stats['n']}")

    out_path = CACHE_DIR / "phase2_external_eval.json"
    out_path.write_text(json.dumps({
        "wild": wild_results,
        "synthetic": synth_results,
        "synthetic_by_label": synth_agg,
        "wild_by_label": wild_agg,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
