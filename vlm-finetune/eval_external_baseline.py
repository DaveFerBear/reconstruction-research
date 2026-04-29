"""
Ablation companion to eval_external.py: same Wild + Synthetic eval, but using
the train_bt.py-style baseline architecture (preresize-mean-pool + 3-layer MLP)
trained on clean corruption pairs. Tests whether the architecture change
matters for cross-distribution transfer (Wild) even if it barely matters
in-distribution (RealCV ablation showed 0.89 vs 0.90).

Usage:
  pai3 && python vlm-finetune/eval_external_baseline.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from encoder_probe import CACHE_DIR, collect_pairs
from eval_external import (
    SYNTHETIC_CACHE,
    WILD_CACHE,
    load_synthetic_pairs,
    load_wild_pairs,
)
from train_baseline_arch import (
    GlobalPairDataset,
    MLPHead,
    bt_loss,
    embed_preresize_global,
    get_or_cache,
)

WILD_BASELINE_CACHE = CACHE_DIR / "uiclip_preresize_global_wild.pt"
SYNTH_BASELINE_CACHE = CACHE_DIR / "uiclip_preresize_global_synthetic.pt"


def get_or_cache_external(paths: list[str], cache_path: Path) -> dict[str, torch.Tensor]:
    cached: dict[str, torch.Tensor] = {}
    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
    missing = [Path(p) for p in paths if p not in cached]
    if missing:
        print(f"  embedding {len(missing)} new images for {cache_path.name}")
        new = embed_preresize_global(missing)
        cached.update(new)
        torch.save(cached, cache_path)
    return cached


@torch.no_grad()
def score_pair(model: MLPHead, embeddings: dict, path_a: str, path_b: str, device: str) -> tuple[float, float]:
    a = embeddings[path_a].unsqueeze(0).to(device)
    b = embeddings[path_b].unsqueeze(0).to(device)
    return float(model(a).item()), float(model(b).item())


def evaluate_external(model: MLPHead, pairs: list, embeddings: dict, device: str) -> dict:
    model.eval()
    correct = 0
    by_label: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path_aes, path_neg, label in pairs:
        s_aes, s_neg = score_pair(model, embeddings, path_aes, path_neg, device)
        is_correct = int(s_aes > s_neg)
        correct += is_correct
        by_label[label][0] += is_correct
        by_label[label][1] += 1
    return {
        "overall": correct / len(pairs) if pairs else 0.0,
        "n": len(pairs),
        "by_label": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in by_label.items()},
    }


def train_full_baseline(embeddings: dict, epochs: int, batch_size: int, lr: float, wd: float, hidden: int, device: str) -> MLPHead:
    pairs = collect_pairs()
    train_loader = DataLoader(GlobalPairDataset(pairs, embeddings), batch_size=batch_size, shuffle=True)
    model = MLPHead(dim=512, hidden_dim=hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_b = 0
        for clean, corr, _ in train_loader:
            clean, corr = clean.to(device), corr.to(device)
            s_clean = model(clean)
            s_corr = model(corr)
            loss = bt_loss(s_clean, s_corr)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_b += 1
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"  epoch {epoch:3d}  loss={total_loss / n_b:.3f}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    train_pairs = collect_pairs()
    train_paths = sorted({p.clean_path for p in train_pairs} | {p.corrupted_path for p in train_pairs})
    train_embs = get_or_cache(train_paths)

    wild_pairs = load_wild_pairs()
    synth_pairs = load_synthetic_pairs()
    print(f"Wild:      {len(wild_pairs)} pairs")
    print(f"Synthetic: {len(synth_pairs)} pairs")

    wild_paths = sorted({p[0] for p in wild_pairs} | {p[1] for p in wild_pairs})
    synth_paths = sorted({p[0] for p in synth_pairs} | {p[1] for p in synth_pairs})
    wild_embs = get_or_cache_external(wild_paths, WILD_BASELINE_CACHE)
    synth_embs = get_or_cache_external(synth_paths, SYNTH_BASELINE_CACHE)

    wild_results: list[dict] = []
    synth_results: list[dict] = []
    for seed in range(args.seeds):
        print(f"\n=== Training baseline-arch model on all {len(train_pairs)} pairs, seed={seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = train_full_baseline(train_embs, args.epochs, args.batch_size, args.lr, args.wd, args.hidden, device)
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
    print("BASELINE ARCH (preresize-mean-pool + 3-MLP, train_bt.py-style)")

    def fmt_ci(arr: np.ndarray) -> str:
        n = len(arr)
        if n <= 1:
            return f"{arr[0]:.3f}"
        mean = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(n)
        return f"{mean:.3f} ± {arr.std(ddof=1):.3f} (95% CI [{mean - 1.96 * se:.3f}, {mean + 1.96 * se:.3f}])"

    print(f"Synthetic ({synth_results[0]['n']} pairs):  {fmt_ci(synth_acc)}")
    print(f"Wild      ({wild_results[0]['n']} pairs): {fmt_ci(wild_acc)}")

    out_path = CACHE_DIR / "phase2_baseline_arch_external.json"
    out_path.write_text(json.dumps({"wild": wild_results, "synthetic": synth_results}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
