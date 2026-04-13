"""
Build an aesthetic / non-aesthetic pair dataset from the reconstruction-research
edit pipeline outputs.

Aesthetic side  = datasets/specs/{template_id}/render.png         (the untouched original)
Non-aesthetic   = edits/{edit_id}/render.png                      (an agent's edit that
                                                                   scored worse than the
                                                                   original per Gemini)

Outputs (under vlm-finetune/dataset/):
  pairs.jsonl        - one JSON object per pair with absolute + relative paths,
                       scores, instruction, agent, template_id, edit_type, etc.
  pairs.csv          - same data in CSV form for quick inspection
  pairs_hard.jsonl   - subset with score_diff <= -10 (clearer negatives)
  images/            - (optional, --copy) a flat copy of every referenced PNG
                       renamed to {pair_id}_{aesthetic|non_aesthetic}.png
  README.md          - dataset card

Usage:
  python build_dataset.py                       # default: metadata only, threshold <0
  python build_dataset.py --threshold -5        # stricter negatives
  python build_dataset.py --copy                # also materialize images/
  python build_dataset.py --include-positives   # also emit (orig, improved-edit) flipped pairs
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

# -------- paths (script lives in reconstruction-research/vlm-finetune/) --------
REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = REPO_ROOT / "datasets" / "specs"
EDITS_DIR = REPO_ROOT / "edits"
SCORES_CSV = REPO_ROOT / "notebooks" / "edit_results_scored.csv"
OUT_DIR = Path(__file__).resolve().parent / "dataset"


def load_scored() -> pd.DataFrame:
    df = pd.read_csv(SCORES_CSV)
    df = df[df["status"] == "success"].dropna(subset=["score_diff"]).copy()
    df["real_template"] = df["edit_id"].str.extract(r"^(.+?)_t\d+_")[0]
    df["edit_num"] = df["edit_id"].str.extract(r"_t(\d+)_")[0].astype(int)
    df["aesthetic_path"] = df["real_template"].apply(
        lambda t: SPECS_DIR / t / "render.png"
    )
    df["non_aesthetic_path"] = df["edit_id"].apply(
        lambda e: EDITS_DIR / e / "render.png"
    )
    before = len(df)
    df = df[
        df["aesthetic_path"].map(Path.exists)
        & df["non_aesthetic_path"].map(Path.exists)
    ].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  dropped {dropped} rows missing render.png", file=sys.stderr)
    return df


def build_pairs(df: pd.DataFrame, threshold: float, include_positives: bool) -> pd.DataFrame:
    # Strict inequality: threshold=0 means score_diff < 0 (actual regressions, no ties).
    regressions = df[df["score_diff"] < threshold].copy() if threshold != 0.0 \
        else df[df["score_diff"] < 0].copy()
    regressions["pair_type"] = "regression"
    regressions["aesthetic_score"] = regressions["original_score"]
    regressions["non_aesthetic_score"] = regressions["aesthetic_score"] - (
        -regressions["score_diff"]  # score_diff = aesthetic - original; invert
    )
    # Recompute cleanly from source columns to avoid sign confusion:
    regressions["aesthetic_score"] = regressions["original_score"]
    regressions["non_aesthetic_score"] = regressions["aesthetic_score"] + regressions["score_diff"]

    frames = [regressions]
    if include_positives:
        # Flipped pairs: agent produced something the critic liked MORE than the original.
        # For our contrastive setup this means the original is the "non-aesthetic" side.
        pos = df[df["score_diff"] > 0].copy()
        pos["pair_type"] = "improvement_flipped"
        # Swap sides so "aesthetic" is always the higher-scored image.
        pos = pos.rename(columns={
            "aesthetic_path": "non_aesthetic_path",
            "non_aesthetic_path": "aesthetic_path",
        })
        pos["aesthetic_score"] = pos["original_score"] + pos["score_diff"]
        pos["non_aesthetic_score"] = pos["original_score"]
        frames.append(pos)

    out = pd.concat(frames, ignore_index=True)
    out["score_gap"] = out["aesthetic_score"] - out["non_aesthetic_score"]
    out["pair_id"] = (
        out["edit_id"] + "__" + out["pair_type"]
    )
    return out


def write_outputs(pairs: pd.DataFrame, copy_images: bool) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    images_dir = OUT_DIR / "images"
    if copy_images:
        images_dir.mkdir(exist_ok=True)

    records = []
    hard_records = []
    for _, row in pairs.iterrows():
        pair_id = row["pair_id"]
        aes_src = Path(row["aesthetic_path"])
        neg_src = Path(row["non_aesthetic_path"])
        if copy_images:
            aes_dst = images_dir / f"{pair_id}__aesthetic.png"
            neg_dst = images_dir / f"{pair_id}__non_aesthetic.png"
            if not aes_dst.exists():
                shutil.copy2(aes_src, aes_dst)
            if not neg_dst.exists():
                shutil.copy2(neg_src, neg_dst)
            aes_ref = str(aes_dst.relative_to(OUT_DIR))
            neg_ref = str(neg_dst.relative_to(OUT_DIR))
        else:
            aes_ref = str(aes_src.relative_to(REPO_ROOT))
            neg_ref = str(neg_src.relative_to(REPO_ROOT))

        rec = {
            "pair_id": pair_id,
            "pair_type": row["pair_type"],
            "template_id": row["real_template"],
            "edit_type_idx": int(row["edit_num"]),
            "edit_type_name": row.get("template_name"),
            "agent": row["agent"],
            "model": row["model"],
            "instruction": row["instruction"],
            "aesthetic_path": aes_ref,
            "non_aesthetic_path": neg_ref,
            "aesthetic_score": float(row["aesthetic_score"]),
            "non_aesthetic_score": float(row["non_aesthetic_score"]),
            "score_gap": float(row["score_gap"]),
            "edit_quality_score": (
                float(row["edit_quality_score"])
                if pd.notna(row.get("edit_quality_score"))
                else None
            ),
        }
        records.append(rec)
        if rec["score_gap"] >= 10:
            hard_records.append(rec)

    with (OUT_DIR / "pairs.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with (OUT_DIR / "pairs_hard.jsonl").open("w") as f:
        for r in hard_records:
            f.write(json.dumps(r) + "\n")

    pd.DataFrame(records).to_csv(OUT_DIR / "pairs.csv", index=False)

    readme = f"""# Aesthetic / Non-aesthetic Pair Dataset

Generated by `build_dataset.py` from `notebooks/edit_results_scored.csv`.

**Semantics.** In every row:
- `aesthetic_path` = the *higher*-Gemini-scored image
- `non_aesthetic_path` = the *lower*-Gemini-scored image
- `score_gap = aesthetic_score - non_aesthetic_score` (always >= 0)

**pair_type = regression** (primary): original template render vs. an agent edit
that the critic scored worse. This is the clean "aesthetic = intact design,
non-aesthetic = broken/regressed design" supervision.

**pair_type = improvement_flipped** (only when built with `--include-positives`):
an agent edit that scored *better* than the original. The agent's output becomes
the aesthetic side and the original becomes the non-aesthetic side. Use with
caution — the critic was only asked for an aesthetic score, not whether the edit
instruction was respected, so these can be noisy.

**Counts**
- Total pairs: {len(records)}
- Hard pairs (|Δ| >= 10): {len(hard_records)}

Paths in `pairs.jsonl` are relative to `reconstruction-research/` unless
`--copy` was used, in which case they live under `dataset/images/`.
"""
    (OUT_DIR / "README.md").write_text(readme)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="keep regression pairs where score_diff <= threshold (default: 0)")
    ap.add_argument("--copy", action="store_true",
                    help="copy images into dataset/images/ instead of referencing source paths")
    ap.add_argument("--include-positives", action="store_true",
                    help="also include improvement-flipped pairs (agent > original)")
    args = ap.parse_args()

    print(f"Loading scores from {SCORES_CSV}")
    df = load_scored()
    print(f"  {len(df)} scored successful edits with renders present")

    pairs = build_pairs(df, threshold=args.threshold, include_positives=args.include_positives)
    print(f"Built {len(pairs)} pairs "
          f"(regression={int((pairs['pair_type']=='regression').sum())}, "
          f"positive={int((pairs['pair_type']=='improvement_flipped').sum())})")

    hard = int((pairs['score_gap'] >= 10).sum())
    very_hard = int((pairs['score_gap'] >= 20).sum())
    print(f"  |Δ|>=10: {hard}    |Δ|>=20: {very_hard}")
    print(f"  unique templates: {pairs['real_template'].nunique()}")
    print(f"  agents: {pairs['agent'].value_counts().to_dict()}")

    print(f"Writing to {OUT_DIR}" + (" (copying images)" if args.copy else " (metadata only)"))
    write_outputs(pairs, copy_images=args.copy)
    print("Done.")


if __name__ == "__main__":
    main()
