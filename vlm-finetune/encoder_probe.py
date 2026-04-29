"""
Phase 0 encoder probe: do candidate frozen encoders carry the failure-mode signal?

For each (encoder, pooling, failure-mode), computes the separation ratio

    r = median(δ_corruption) / median(δ_control)

where δ is L2 distance between encoded representations of an image pair,
the corruption set is (clean_original, single-mode-corrupted) pairs from
evals/data_real/, and the control set is random clean-clean pairs drawn from
different specs (the inter-design noise floor).

r ≪ 1  ⇒ corrupted designs look more similar to their clean original than two
        random designs look to each other — the encoder is blind to that mode
        at this pooling, and no head on top will recover the signal.
r ≈ 1  ⇒ corruption signal is comparable to inter-design noise — a head might
        learn it given enough data.
r > 1  ⇒ corruption pushes the embedding further than design-to-design
        differences — easy regime.

Outputs:
  cache/probe_results.json  — full numerics
  cache/probe_heatmap.png   — rows = (encoder × pooling), cols = modes

Usage:
  pai3 && python vlm-finetune/encoder_probe.py
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = REPO_ROOT / "datasets" / "specs"
DATA_REAL_DIR = REPO_ROOT / "evals" / "data_real"
CACHE_DIR = Path(__file__).resolve().parent / "cache"


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
    crops: list[Image.Image] = []
    for y in range(0, h - sq + 1, step if h > w else sq):
        for x in range(0, w - sq + 1, step if w > h else sq):
            crops.append(image.crop((x, y, x + sq, y + sq)))
    return crops


def native_grid_crops(image: Image.Image, size: int) -> list[Image.Image]:
    """Tile the image into `size`×`size` crops at *native* resolution (no preresize).

    Crops the image into a regular grid of size×size tiles. Last row/column may
    overlap the previous tile if the image isn't a multiple of `size` (so we
    don't drop the final strip). Crop layout depends only on canvas size, so
    aligned pairs (clean, corrupted) yield identical crop counts and positions.
    """
    w, h = image.size
    if w < size or h < size:
        # Tiny canvas — fall back to one center crop at native res.
        return [image.crop((0, 0, min(w, size), min(h, size)))]
    xs = list(range(0, w - size + 1, size))
    if xs[-1] + size < w:
        xs.append(w - size)
    ys = list(range(0, h - size + 1, size))
    if ys[-1] + size < h:
        ys.append(h - size)
    crops: list[Image.Image] = []
    for y in ys:
        for x in xs:
            crops.append(image.crop((x, y, x + size, y + size)))
    return crops


@dataclass(frozen=True)
class CorruptionPair:
    spec_id: str
    mode: str
    clean_path: Path
    corrupted_path: Path


def collect_pairs() -> list[CorruptionPair]:
    pairs: list[CorruptionPair] = []
    for spec_dir in sorted(DATA_REAL_DIR.iterdir()):
        if not spec_dir.is_dir():
            continue
        clean = SPECS_DIR / spec_dir.name / "render.png"
        if not clean.exists():
            continue
        for mode_dir in sorted(spec_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            corrupted = mode_dir / "render.png"
            if not corrupted.exists():
                continue
            pairs.append(CorruptionPair(
                spec_id=spec_dir.name,
                mode=mode_dir.name,
                clean_path=clean,
                corrupted_path=corrupted,
            ))
    return pairs


class Probe:
    """A frozen encoder. encode_batch(paths) returns one fingerprint dict per image.

    Each dict contains:
      'global': (d,) np.ndarray  — global L2-normalized embedding
      'crops':  (n_crops, d) np.ndarray  — L2-normalized per-crop embeddings,
                                           in the same spatial order across pairs
                                           (slide_windows is deterministic on canvas size)
    """
    name: str

    def encode_batch(self, paths: list[Path]) -> list[dict[str, np.ndarray]]:
        raise NotImplementedError


class UICLIPProbe(Probe):
    UICLIP_MODEL = "biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs"
    UICLIP_PROCESSOR = "openai/clip-vit-base-patch32"
    IMG_SIZE = 224

    def __init__(self, batch_size: int = 16):
        from transformers import CLIPModel, CLIPProcessor

        self.name = "ui-clip"
        self.batch_size = batch_size
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(self.UICLIP_MODEL).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.UICLIP_PROCESSOR)

    @torch.no_grad()
    def _embed_crops(self, crops: list[Image.Image]) -> np.ndarray:
        """Embed a list of PIL crops, return (n_crops, d) L2-normalized array."""
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        inputs = self.processor(images=crops, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    @torch.no_grad()
    def encode_batch(self, paths: list[Path]) -> list[dict[str, np.ndarray]]:
        out: list[dict[str, np.ndarray]] = []
        for i in range(0, len(paths), self.batch_size):
            batch = paths[i : i + self.batch_size]
            # Two crop strategies per image, embedded together to minimize forward passes.
            preresize_crops_per_img: list[list[Image.Image]] = []
            native_crops_per_img: list[list[Image.Image]] = []
            for p in batch:
                img = Image.open(p).convert("RGB")
                preresize_crops_per_img.append(slide_windows(img, self.IMG_SIZE))
                native_crops_per_img.append(native_grid_crops(img, self.IMG_SIZE))

            # Flatten and embed in two passes (different crop counts, can't trivially share).
            flat_pre: list[Image.Image] = []
            flat_pre_idx: list[int] = []
            for j, cs in enumerate(preresize_crops_per_img):
                flat_pre.extend(cs)
                flat_pre_idx.extend([j] * len(cs))
            flat_native: list[Image.Image] = []
            flat_native_idx: list[int] = []
            for j, cs in enumerate(native_crops_per_img):
                flat_native.extend(cs)
                flat_native_idx.extend([j] * len(cs))

            pre_feats = self._embed_crops(flat_pre)
            native_feats = self._embed_crops(flat_native)
            pre_idx = np.array(flat_pre_idx)
            native_idx = np.array(flat_native_idx)

            for j in range(len(batch)):
                pre_block = pre_feats[pre_idx == j]
                native_block = native_feats[native_idx == j]
                global_vec = pre_block.mean(axis=0)
                global_vec = global_vec / (np.linalg.norm(global_vec) + 1e-12)
                out.append({
                    "global": global_vec,
                    "crops": pre_block,
                    "native_crops": native_block,
                })
            done = min(i + self.batch_size, len(paths))
            print(f"  {self.name}: encoded {done}/{len(paths)}", end="\r")
        print()
        return out


def delta_global(a: dict, b: dict) -> float:
    return float(np.linalg.norm(a["global"] - b["global"]))


def _aligned_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = min(len(A), len(B))
    if n == 0:
        return np.array([0.0])
    return np.linalg.norm(A[:n] - B[:n], axis=1)


def delta_crop_mean(a: dict, b: dict) -> float:
    return float(_aligned_distances(a["crops"], b["crops"]).mean())


def delta_crop_max(a: dict, b: dict) -> float:
    return float(_aligned_distances(a["crops"], b["crops"]).max())


def delta_native_mean(a: dict, b: dict) -> float:
    return float(_aligned_distances(a["native_crops"], b["native_crops"]).mean())


def delta_native_max(a: dict, b: dict) -> float:
    return float(_aligned_distances(a["native_crops"], b["native_crops"]).max())


METRICS = {
    "global": delta_global,
    "crops_mean": delta_crop_mean,
    "crops_max": delta_crop_max,
    "native_mean": delta_native_mean,
    "native_max": delta_native_max,
}


def run_probe(probe: Probe, pairs: list[CorruptionPair], n_control: int, seed: int) -> dict:
    paths = sorted({p.clean_path for p in pairs} | {p.corrupted_path for p in pairs})
    print(f"  {probe.name}: encoding {len(paths)} unique images")
    embeddings = dict(zip(paths, probe.encode_batch(paths)))

    pair_deltas: dict[str, list[tuple[str, float]]] = {m: [] for m in METRICS}
    for pair in pairs:
        a = embeddings[pair.clean_path]
        b = embeddings[pair.corrupted_path]
        for m, fn in METRICS.items():
            pair_deltas[m].append((pair.mode, fn(a, b)))

    rng = random.Random(seed)
    clean_paths = sorted({p.clean_path for p in pairs})
    spec_of = {p.clean_path: p.spec_id for p in pairs}
    control_pairs: list[tuple[Path, Path]] = []
    while len(control_pairs) < n_control:
        a, b = rng.sample(clean_paths, 2)
        if spec_of[a] != spec_of[b]:
            control_pairs.append((a, b))
    control_deltas: dict[str, list[float]] = {m: [] for m in METRICS}
    for a, b in control_pairs:
        ea, eb = embeddings[a], embeddings[b]
        for m, fn in METRICS.items():
            control_deltas[m].append(fn(ea, eb))

    results: dict = {"probe": probe.name, "metrics": {}}
    for m in METRICS:
        per_mode: dict[str, list[float]] = defaultdict(list)
        for mode, d in pair_deltas[m]:
            per_mode[mode].append(d)
        ctl_med = float(np.median(control_deltas[m]))
        mode_table = {}
        for mode, ds in per_mode.items():
            corr_med = float(np.median(ds))
            mode_table[mode] = {
                "n": len(ds),
                "corruption_median": corr_med,
                "corruption_mean": float(np.mean(ds)),
                "r": corr_med / ctl_med if ctl_med > 0 else float("inf"),
            }
        results["metrics"][m] = {
            "control_median": ctl_med,
            "control_mean": float(np.mean(control_deltas[m])),
            "n_control_pairs": len(control_pairs),
            "by_mode": mode_table,
        }
    return results


def render_heatmap(all_results: list[dict], modes: list[str], out_path: Path) -> None:
    rows: list[list[float]] = []
    row_labels: list[str] = []
    for res in all_results:
        for metric_name, metric_block in res["metrics"].items():
            row = [metric_block["by_mode"].get(m, {}).get("r", float("nan")) for m in modes]
            rows.append(row)
            row_labels.append(f"{res['probe']} / {metric_name}")
    arr = np.array(rows)

    vmax = max(2.0, float(np.nanmax(arr)) if np.any(~np.isnan(arr)) else 2.0)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(modes) + 2), 0.55 * len(rows) + 2.0))
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(len(rows)):
        for j in range(len(modes)):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("r = median(δ_corruption) / median(δ_control)\n(< 1 = encoder blind; > 1 = corruption > inter-design noise)")
    ax.set_title("Phase 0: encoder separation ratio per failure mode")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-control", type=int, default=200, help="Random clean-clean control pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)
    pairs = collect_pairs()
    print(f"Collected {len(pairs)} (clean, corrupted) pairs across {len({p.spec_id for p in pairs})} specs")
    by_mode: dict[str, int] = defaultdict(int)
    for p in pairs:
        by_mode[p.mode] += 1
    for m, n in sorted(by_mode.items(), key=lambda kv: -kv[1]):
        print(f"  {m:25s} {n:4d}")
    modes = sorted(by_mode.keys())

    probes: list[Probe] = [UICLIPProbe(batch_size=args.batch_size)]
    all_results: list[dict] = []
    for probe in probes:
        print(f"\n=== Probe: {probe.name} ===")
        res = run_probe(probe, pairs, n_control=args.n_control, seed=args.seed)
        all_results.append(res)

        print(f"\n  control medians: " + ", ".join(
            f"{m}={res['metrics'][m]['control_median']:.4f}" for m in METRICS
        ))
        for m in METRICS:
            print(f"\n  metric: {m}")
            print(f"    control median: {res['metrics'][m]['control_median']:.4f}")
            mode_block = res['metrics'][m]['by_mode']
            for mode in sorted(mode_block, key=lambda k: -mode_block[k]['n']):
                row = mode_block[mode]
                print(f"    {mode:25s} n={row['n']:3d}  median δ={row['corruption_median']:.4f}  r={row['r']:.3f}")

    out_json = CACHE_DIR / "probe_results.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_json}")

    render_heatmap(all_results, modes, CACHE_DIR / "probe_heatmap.png")


if __name__ == "__main__":
    main()
