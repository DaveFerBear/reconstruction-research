"""
Bradley-Terry reward model on frozen UI-CLIP embeddings.

Phase 1 baseline: can a small MLP on top of a frozen CLIP backbone
distinguish aesthetic from non-aesthetic designs?

Usage:
  pai3 && python vlm-finetune/train_bt.py                    # full run
  pai3 && python vlm-finetune/train_bt.py --embed-only        # just cache embeddings
  pai3 && python vlm-finetune/train_bt.py --epochs 50 --lr 3e-4 --hidden 256
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ─── paths ───────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
PAIRS_FILE = DATASET_DIR / "pairs_hard.jsonl"  # |Δ|≥10
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# ─── UI-CLIP config ──────────────────────────────────────────────────────────
UICLIP_MODEL = "biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs"
UICLIP_PROCESSOR = "openai/clip-vit-base-patch32"
EMBED_DIM = 512
IMG_SIZE = 224


# ─── embedding extraction ────────────────────────────────────────────────────

def preresize(image: Image.Image, size: int) -> Image.Image:
    aspect = image.width / image.height
    if aspect > 1:
        return image.resize((int(aspect * size), size))
    return image.resize((size, int(size / aspect)))


def slide_windows(image: Image.Image, size: int) -> list[Image.Image]:
    image = preresize(image, size)
    w, h = image.size
    sq = min(w, h)
    longer = max(w, h)
    n_steps = (longer + sq - 1) // sq
    step = (longer - sq) // (n_steps - 1) if n_steps > 1 else sq
    crops = []
    for y in range(0, h - sq + 1, step if h > w else sq):
        for x in range(0, w - sq + 1, step if w > h else sq):
            crops.append(image.crop((x, y, x + sq, y + sq)))
    return crops


@torch.no_grad()
def extract_embeddings(image_paths: list[str], batch_size: int = 32) -> dict[str, torch.Tensor]:
    """Extract UI-CLIP image embeddings for a list of image paths."""
    from transformers import CLIPProcessor, CLIPModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained(UICLIP_MODEL).eval().to(device)
    processor = CLIPProcessor.from_pretrained(UICLIP_PROCESSOR)

    embeddings = {}
    total = len(image_paths)

    for i in range(0, total, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        all_crops = []
        indices = []  # maps each crop back to its image index in batch

        for j, p in enumerate(batch_paths):
            img = Image.open(p).convert("RGB")
            crops = slide_windows(img, IMG_SIZE)
            all_crops.extend(crops)
            indices.extend([j] * len(crops))

        inputs = processor(images=all_crops, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)  # (n_crops, 512)

        # average-pool crops per image
        indices_t = torch.tensor(indices, device=feats.device)
        for j, p in enumerate(batch_paths):
            mask = indices_t == j
            emb = feats[mask].mean(dim=0)
            emb = emb / emb.norm()  # L2-normalize
            embeddings[p] = emb.cpu()

        done = min(i + batch_size, total)
        print(f"  embedded {done}/{total} images", end="\r")

    print()
    return embeddings


def get_or_cache_embeddings(image_paths: list[str], batch_size: int = 32) -> dict[str, torch.Tensor]:
    """Load cached embeddings or extract and cache them."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "uiclip_embeddings.pt"

    if cache_file.exists():
        print(f"Loading cached embeddings from {cache_file}")
        cached = torch.load(cache_file, weights_only=True)
        # Check if all needed paths are cached
        missing = [p for p in image_paths if p not in cached]
        if not missing:
            return cached
        print(f"  {len(missing)} new images to embed")
    else:
        cached = {}
        missing = image_paths

    new_embeds = extract_embeddings(missing, batch_size=batch_size)
    cached.update(new_embeds)
    torch.save(cached, cache_file)
    print(f"Saved {len(cached)} embeddings to {cache_file}")
    return cached


# ─── dataset ─────────────────────────────────────────────────────────────────

def load_pairs() -> list[dict]:
    pairs = []
    with open(PAIRS_FILE) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def split_by_template(pairs: list[dict], val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42):
    """Split pairs into train/val/test by template_id so no template leaks."""
    templates = sorted(set(p["template_id"] for p in pairs))
    rng = np.random.RandomState(seed)
    rng.shuffle(templates)

    n = len(templates)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_t = set(templates[:n_test])
    val_t = set(templates[n_test : n_test + n_val])
    train_t = set(templates[n_test + n_val :])

    train = [p for p in pairs if p["template_id"] in train_t]
    val = [p for p in pairs if p["template_id"] in val_t]
    test = [p for p in pairs if p["template_id"] in test_t]

    print(f"Split: {len(train_t)} train templates ({len(train)} pairs), "
          f"{len(val_t)} val ({len(val)}), {len(test_t)} test ({len(test)})")
    return train, val, test


class PairDataset(Dataset):
    def __init__(self, pairs: list[dict], embeddings: dict[str, torch.Tensor]):
        self.pairs = pairs
        self.embeddings = embeddings

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        aes_path = str(DATASET_DIR / p["aesthetic_path"])
        neg_path = str(DATASET_DIR / p["non_aesthetic_path"])
        return (
            self.embeddings[aes_path],
            self.embeddings[neg_path],
            torch.tensor(p["score_gap"], dtype=torch.float32),
        )


# ─── model ───────────────────────────────────────────────────────────────────

class RewardHead(nn.Module):
    """Small MLP that maps a CLIP embedding to a scalar aesthetic score."""

    def __init__(self, input_dim: int = EMBED_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─── training ────────────────────────────────────────────────────────────────

def bt_loss(score_chosen: torch.Tensor, score_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise loss: -log σ(s_chosen - s_rejected)."""
    return -F.logsigmoid(score_chosen - score_rejected).mean()


def pairwise_accuracy(model: RewardHead, loader: DataLoader, device: str) -> tuple[float, float]:
    """Compute pairwise accuracy and mean score gap correlation."""
    model.eval()
    correct = 0
    total = 0
    pred_diffs = []
    true_gaps = []

    with torch.no_grad():
        for emb_aes, emb_neg, gap in loader:
            emb_aes, emb_neg = emb_aes.to(device), emb_neg.to(device)
            s_aes = model(emb_aes)
            s_neg = model(emb_neg)
            correct += (s_aes > s_neg).sum().item()
            total += len(gap)
            pred_diffs.extend((s_aes - s_neg).cpu().tolist())
            true_gaps.extend(gap.tolist())

    acc = correct / total if total > 0 else 0.0
    # Spearman-like: correlation between predicted diff and true score gap
    pred_arr = np.array(pred_diffs)
    true_arr = np.array(true_gaps)
    if len(pred_arr) > 1 and np.std(pred_arr) > 0 and np.std(true_arr) > 0:
        corr = np.corrcoef(pred_arr, true_arr)[0, 1]
    else:
        corr = 0.0
    return acc, corr


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load pairs and collect image paths
    pairs = load_pairs()
    print(f"Loaded {len(pairs)} hard pairs (|Δ|≥10)")

    all_image_paths = set()
    for p in pairs:
        all_image_paths.add(str(DATASET_DIR / p["aesthetic_path"]))
        all_image_paths.add(str(DATASET_DIR / p["non_aesthetic_path"]))
    all_image_paths = sorted(all_image_paths)
    print(f"Unique images: {len(all_image_paths)}")

    # Extract / load embeddings
    embeddings = get_or_cache_embeddings(all_image_paths, batch_size=args.batch_embed)

    if args.embed_only:
        print("--embed-only: done.")
        return

    # Split
    train_pairs, val_pairs, test_pairs = split_by_template(pairs)

    train_ds = PairDataset(train_pairs, embeddings)
    val_ds = PairDataset(val_pairs, embeddings)
    test_ds = PairDataset(test_pairs, embeddings)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Model
    model = RewardHead(input_dim=EMBED_DIM, hidden_dim=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"RewardHead: {param_count:,} parameters (hidden={args.hidden})")

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for emb_aes, emb_neg, gap in train_loader:
            emb_aes, emb_neg = emb_aes.to(device), emb_neg.to(device)
            s_aes = model(emb_aes)
            s_neg = model(emb_neg)
            loss = bt_loss(s_aes, s_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        train_acc, train_corr = pairwise_accuracy(model, train_loader, device)
        val_acc, val_corr = pairwise_accuracy(model, val_loader, device)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  loss={avg_loss:.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                  f"train_corr={train_corr:.3f}  val_corr={val_corr:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), CACHE_DIR / "best_reward_head.pt")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (best val_acc={best_val_acc:.3f} at epoch {best_epoch})")
            break

    # Final evaluation on test set
    model.load_state_dict(torch.load(CACHE_DIR / "best_reward_head.pt", weights_only=True))
    model.to(device)
    test_acc, test_corr = pairwise_accuracy(model, test_loader, device)
    val_acc, val_corr = pairwise_accuracy(model, val_loader, device)

    print("\n" + "=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Val  pairwise accuracy: {val_acc:.3f}  |  score-gap correlation: {val_corr:.3f}")
    print(f"Test pairwise accuracy: {test_acc:.3f}  |  score-gap correlation: {test_corr:.3f}")
    print("=" * 60)

    # Calibration breakdown: accuracy by score-gap bucket
    print("\nCalibration (test set) — accuracy by score gap bucket:")
    model.eval()
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    with torch.no_grad():
        for emb_aes, emb_neg, gap in test_loader:
            emb_aes, emb_neg = emb_aes.to(device), emb_neg.to(device)
            s_aes = model(emb_aes)
            s_neg = model(emb_neg)
            correct = (s_aes > s_neg).cpu()
            for c, g in zip(correct, gap):
                if g < 15:
                    b = "10-14"
                elif g < 20:
                    b = "15-19"
                elif g < 30:
                    b = "20-29"
                else:
                    b = "30+"
                buckets[b]["correct"] += c.item()
                buckets[b]["total"] += 1

    for b in ["10-14", "15-19", "20-29", "30+"]:
        if buckets[b]["total"] > 0:
            acc = buckets[b]["correct"] / buckets[b]["total"]
            print(f"  |Δ| {b:>5s}: {acc:.3f}  ({buckets[b]['total']} pairs)")

    # Save config for reproducibility
    config = {
        "model": "UI-CLIP frozen + MLP reward head",
        "clip_model": UICLIP_MODEL,
        "embed_dim": EMBED_DIM,
        "hidden_dim": args.hidden,
        "lr": args.lr,
        "weight_decay": args.wd,
        "epochs_run": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_corr": test_corr,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "n_test_pairs": len(test_pairs),
        "param_count": param_count,
    }
    with open(CACHE_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nModel saved to {CACHE_DIR / 'best_reward_head.pt'}")
    print(f"Config saved to {CACHE_DIR / 'config.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-only", action="store_true", help="Only extract and cache embeddings, skip training")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    ap.add_argument("--hidden", type=int, default=128, help="MLP hidden dim")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--batch-embed", type=int, default=16, help="Batch size for embedding extraction")
    ap.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
