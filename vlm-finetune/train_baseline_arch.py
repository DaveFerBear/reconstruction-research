"""
Ablation: train_bt.py-style architecture (preresize → slide-windows →
mean-pool → 3-layer MLP) on the *clean* corruption pairs from evals/data_real/.

Holds everything constant except the encoder/pooling, so the delta vs.
`vlm-finetune/train.py` (Phase 1) isolates the contribution of native-res +
attention pool vs. the fix from clean labels alone.

Usage:
  pai3 && python vlm-finetune/train_baseline_arch.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from encoder_probe import (
    CACHE_DIR,
    CorruptionPair,
    collect_pairs,
    slide_windows,  # the preresize-then-windows function from train_bt.py
)
from train import EMBED_DIM, IMG_SIZE, UICLIP_MODEL, UICLIP_PROCESSOR

PRERESIZE_CACHE = CACHE_DIR / "uiclip_preresize_global.pt"


@torch.no_grad()
def embed_preresize_global(paths: list[Path], batch_size: int = 32) -> dict[str, torch.Tensor]:
    """Mirror train_bt.py: preresize, slide windows at 224, mean-pool crops, L2 normalize."""
    from transformers import CLIPModel, CLIPProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  loading UI-CLIP onto {device}")
    model = CLIPModel.from_pretrained(UICLIP_MODEL).eval().to(device)
    processor = CLIPProcessor.from_pretrained(UICLIP_PROCESSOR)

    out: dict[str, torch.Tensor] = {}
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        flat: list[Image.Image] = []
        idx: list[int] = []
        for j, p in enumerate(batch):
            img = Image.open(p).convert("RGB")
            crops = slide_windows(img, IMG_SIZE)
            flat.extend(crops)
            idx.extend([j] * len(crops))
        if not flat:
            continue
        inputs = processor(images=flat, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        idx_t = torch.tensor(idx, device=feats.device)
        for j, p in enumerate(batch):
            mean_vec = feats[idx_t == j].mean(dim=0)
            mean_vec = mean_vec / (mean_vec.norm() + 1e-12)
            out[str(p)] = mean_vec.cpu()
        done = min(i + batch_size, len(paths))
        print(f"  embedded {done}/{len(paths)}", end="\r")
    print()
    return out


def get_or_cache(paths: list[Path]) -> dict[str, torch.Tensor]:
    PRERESIZE_CACHE.parent.mkdir(exist_ok=True)
    cached: dict[str, torch.Tensor] = {}
    if PRERESIZE_CACHE.exists():
        cached = torch.load(PRERESIZE_CACHE, weights_only=True)
    missing = [p for p in paths if str(p) not in cached]
    if missing:
        print(f"  embedding {len(missing)} new images (preresize-global)")
        new = embed_preresize_global(missing)
        cached.update(new)
        torch.save(cached, PRERESIZE_CACHE)
    return cached


class GlobalPairDataset(Dataset):
    def __init__(self, pairs: list[CorruptionPair], embeddings: dict[str, torch.Tensor]):
        self.pairs = pairs
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        return (
            self.embeddings[str(p.clean_path)],
            self.embeddings[str(p.corrupted_path)],
            p.mode,
        )


class MLPHead(nn.Module):
    """3-layer MLP head — same architecture as train_bt.py:183-197 (hidden=128 → hidden → 1)."""

    def __init__(self, dim: int = EMBED_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def bt_loss(s_chosen: torch.Tensor, s_rejected: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(s_chosen - s_rejected).mean()


@torch.no_grad()
def evaluate(model: MLPHead, loader: DataLoader, device: str) -> dict:
    model.eval()
    correct = 0
    total = 0
    by_mode_correct: dict[str, int] = defaultdict(int)
    by_mode_total: dict[str, int] = defaultdict(int)
    for clean, corr, modes in loader:
        clean, corr = clean.to(device), corr.to(device)
        s_clean = model(clean)
        s_corr = model(corr)
        wins = (s_clean > s_corr).cpu().tolist()
        for w, m in zip(wins, modes):
            correct += int(w)
            total += 1
            by_mode_correct[m] += int(w)
            by_mode_total[m] += 1
    return {
        "overall": correct / total if total else 0.0,
        "by_mode": {m: by_mode_correct[m] / by_mode_total[m] for m in by_mode_total},
        "n": total,
    }


def split_by_spec(pairs: list[CorruptionPair], n_folds: int, seed: int):
    spec_ids = sorted({p.spec_id for p in pairs})
    rng = np.random.RandomState(seed)
    rng.shuffle(spec_ids)
    folds = [spec_ids[i::n_folds] for i in range(n_folds)]
    for k in range(n_folds):
        test_specs = set(folds[k])
        train_pairs = [p for p in pairs if p.spec_id not in test_specs]
        test_pairs = [p for p in pairs if p.spec_id in test_specs]
        yield k, train_pairs, test_pairs


def train_one_fold(train_pairs, test_pairs, embeddings, args, device, log_prefix=""):
    train_loader = DataLoader(GlobalPairDataset(train_pairs, embeddings), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(GlobalPairDataset(test_pairs, embeddings), batch_size=args.batch_size)
    model = MLPHead(dim=EMBED_DIM, hidden_dim=args.hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_b = 0
        for clean, corr, _ in train_loader:
            clean, corr = clean.to(device), corr.to(device)
            s_clean = model(clean)
            s_corr = model(corr)
            loss = bt_loss(s_clean, s_corr)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_b += 1
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            train_metrics = evaluate(model, train_loader, device)
            test_metrics = evaluate(model, test_loader, device)
            print(
                f"  {log_prefix}epoch {epoch:3d}  loss={epoch_loss / n_b:.3f}  "
                f"train={train_metrics['overall']:.3f}  test={test_metrics['overall']:.3f}"
            )
    return evaluate(model, test_loader, device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    pairs = collect_pairs()
    paths = sorted({p.clean_path for p in pairs} | {p.corrupted_path for p in pairs})
    print(f"Loaded {len(pairs)} pairs ({len(paths)} unique images)")
    embeddings = get_or_cache(paths)

    all_results: list[dict] = []
    for seed in range(args.seeds):
        for fold_idx, train_pairs, test_pairs in split_by_spec(pairs, args.folds, seed=42 + seed):
            torch.manual_seed(seed * 1000 + fold_idx)
            print(f"\n=== seed={seed} fold={fold_idx}  train={len(train_pairs)}  test={len(test_pairs)} ===")
            res = train_one_fold(train_pairs, test_pairs, embeddings, args, device, log_prefix=f"  s{seed}f{fold_idx} ")
            res["seed"] = seed
            res["fold"] = fold_idx
            all_results.append(res)
            print(f"  fold result: overall={res['overall']:.3f}")

    overalls = np.array([r["overall"] for r in all_results])
    n = len(overalls)
    se = overalls.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    print("\n" + "=" * 60)
    print("ABLATION: preresize-mean-pool global + 3-layer MLP (train_bt.py architecture)")
    print(f"  pairwise accuracy: mean={overalls.mean():.3f}  sd={overalls.std(ddof=1):.3f}  95% CI [{overalls.mean() - 1.96 * se:.3f}, {overalls.mean() + 1.96 * se:.3f}]")

    by_mode_runs: dict[str, list[float]] = defaultdict(list)
    for r in all_results:
        for mode, acc in r["by_mode"].items():
            by_mode_runs[mode].append(acc)
    print("\nPer-mode (mean ± sd):")
    for mode in sorted(by_mode_runs, key=lambda k: -float(np.mean(by_mode_runs[k]))):
        accs = np.array(by_mode_runs[mode])
        print(f"  {mode:25s}  {accs.mean():.3f} ± {accs.std(ddof=1) if len(accs) > 1 else 0.0:.3f}  (n_runs={len(accs)})")

    out_path = CACHE_DIR / "phase1_baseline_arch_results.json"
    out_path.write_text(json.dumps([{**r, "by_mode": dict(r["by_mode"])} for r in all_results], indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
