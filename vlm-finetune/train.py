"""
Phase 1: scalar Bradley-Terry ranker on native-resolution UI-CLIP per-crop embeddings.

Training data: (clean original, single-mode-corrupted) pairs from evals/data_real/.
Encoder: frozen UI-CLIP, embedding 224² grid crops at *native* canvas resolution
         (no preresize — that was the Phase 0 finding).
Head: attention-pool over the variable-length crop sequence (learnable query,
      multi-head attention) → 2-layer MLP → scalar s(image).
Loss: L = −log σ(s(clean) − s(corrupted)).
Eval: 5-fold CV by spec_id × 3 seeds. Per-fold per-mode pairwise accuracy.

Usage:
  pai3 && python vlm-finetune/train.py
  pai3 && python vlm-finetune/train.py --embed-only       # cache embeddings, skip training
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
    native_grid_crops,
)

UICLIP_MODEL = "biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs"
UICLIP_PROCESSOR = "openai/clip-vit-base-patch32"
EMBED_DIM = 512
IMG_SIZE = 224

NATIVE_CACHE = CACHE_DIR / "uiclip_native_crops.pt"


# ─── encoding (one-time, cached) ─────────────────────────────────────────────

@torch.no_grad()
def embed_native_crops(paths: list[Path], batch_size: int = 16) -> dict[str, torch.Tensor]:
    """For each image path, return an (n_crops, 512) L2-normalized tensor."""
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
            crops = native_grid_crops(img, IMG_SIZE)
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
            out[str(p)] = feats[idx_t == j].cpu()
        done = min(i + batch_size, len(paths))
        print(f"  embedded {done}/{len(paths)}", end="\r")
    print()
    return out


def get_or_cache_embeddings(paths: list[Path]) -> dict[str, torch.Tensor]:
    NATIVE_CACHE.parent.mkdir(exist_ok=True)
    cached: dict[str, torch.Tensor] = {}
    if NATIVE_CACHE.exists():
        print(f"Loading native crop cache from {NATIVE_CACHE}")
        cached = torch.load(NATIVE_CACHE, weights_only=True)
    missing = [p for p in paths if str(p) not in cached]
    if missing:
        print(f"  embedding {len(missing)} new images")
        new = embed_native_crops(missing)
        cached.update(new)
        torch.save(cached, NATIVE_CACHE)
        print(f"  saved {len(cached)} embeddings to {NATIVE_CACHE}")
    else:
        print(f"  all {len(paths)} images cached")
    return cached


# ─── dataset ─────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
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


def pad_collate(batch):
    cleans = [b[0] for b in batch]
    corrs = [b[1] for b in batch]
    modes = [b[2] for b in batch]
    max_len = max(max(c.shape[0] for c in cleans), max(c.shape[0] for c in corrs))

    def pad(seqs):
        B, L = len(seqs), max_len
        out = torch.zeros(B, L, EMBED_DIM)
        mask = torch.zeros(B, L, dtype=torch.bool)
        for i, s in enumerate(seqs):
            n = s.shape[0]
            out[i, :n] = s
            mask[i, :n] = True
        return out, mask

    clean_x, clean_m = pad(cleans)
    corr_x, corr_m = pad(corrs)
    return clean_x, clean_m, corr_x, corr_m, modes


# ─── model ───────────────────────────────────────────────────────────────────

class AttentionPoolHead(nn.Module):
    """Learnable-query attention pool over crop embeddings, then MLP → scalar."""

    def __init__(self, dim: int = EMBED_DIM, n_heads: int = 4, mlp_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        # MultiheadAttention key_padding_mask: True = position is padding (mask out)
        pooled, _ = self.attn(q, x, x, key_padding_mask=~mask)
        return self.mlp(pooled.squeeze(1)).squeeze(-1)


# ─── training ────────────────────────────────────────────────────────────────

def bt_loss(s_chosen: torch.Tensor, s_rejected: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(s_chosen - s_rejected).mean()


@torch.no_grad()
def evaluate(model: AttentionPoolHead, loader: DataLoader, device: str) -> dict:
    model.eval()
    correct = 0
    total = 0
    by_mode_correct: dict[str, int] = defaultdict(int)
    by_mode_total: dict[str, int] = defaultdict(int)
    for clean_x, clean_m, corr_x, corr_m, modes in loader:
        clean_x, clean_m = clean_x.to(device), clean_m.to(device)
        corr_x, corr_m = corr_x.to(device), corr_m.to(device)
        s_clean = model(clean_x, clean_m)
        s_corr = model(corr_x, corr_m)
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
    train_ds = PairDataset(train_pairs, embeddings)
    test_ds = PairDataset(test_pairs, embeddings)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=pad_collate)

    model = AttentionPoolHead(
        dim=EMBED_DIM,
        n_heads=args.n_heads,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
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
            epoch_loss += loss.item()
            n_b += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            train_metrics = evaluate(model, train_loader, device)
            test_metrics = evaluate(model, test_loader, device)
            print(
                f"  {log_prefix}epoch {epoch:3d}  loss={epoch_loss / n_b:.3f}  "
                f"train={train_metrics['overall']:.3f}  test={test_metrics['overall']:.3f}"
            )

    return evaluate(model, test_loader, device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--mlp-hidden", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    pairs = collect_pairs()
    paths = sorted({p.clean_path for p in pairs} | {p.corrupted_path for p in pairs})
    print(f"Loaded {len(pairs)} pairs across {len({p.spec_id for p in pairs})} specs ({len(paths)} unique images)")

    embeddings = get_or_cache_embeddings(paths)
    if args.embed_only:
        return

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
    print(f"Aggregate over {n} runs ({args.seeds} seeds × {args.folds} folds)")
    print(f"  pairwise accuracy: mean={overalls.mean():.3f}  sd={overalls.std(ddof=1):.3f}")
    print(f"  95% CI: [{overalls.mean() - 1.96 * se:.3f}, {overalls.mean() + 1.96 * se:.3f}]")

    by_mode_runs: dict[str, list[float]] = defaultdict(list)
    for r in all_results:
        for mode, acc in r["by_mode"].items():
            by_mode_runs[mode].append(acc)
    print("\nPer-mode pairwise accuracy (mean ± sd across runs):")
    for mode in sorted(by_mode_runs, key=lambda k: -float(np.mean(by_mode_runs[k]))):
        accs = np.array(by_mode_runs[mode])
        print(f"  {mode:25s}  {accs.mean():.3f} ± {accs.std(ddof=1) if len(accs) > 1 else 0.0:.3f}  (n_runs={len(accs)})")

    out_path = CACHE_DIR / "phase1_results.json"
    out_path.write_text(json.dumps(
        [{**r, "by_mode": dict(r["by_mode"])} for r in all_results],
        indent=2,
    ))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
