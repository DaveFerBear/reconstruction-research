"""Apply each corrupter across every spec under datasets/specs/.

For each (spec_id, mode) pair where the corrupter applies, write:
    evals/data_real/<spec_id>/<mode_id>/spec.json
    evals/data_real/<spec_id>/<mode_id>/corruption.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from evals.corrupters import CORRUPTERS

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "datasets" / "specs"
OUT_DIR = Path(__file__).parent / "data_real"


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrupt real specs to induce failure modes")
    parser.add_argument("--spec", help="Process only this spec id (default: all)")
    parser.add_argument("--mode", help="Apply only this corrupter id (default: all)")
    parser.add_argument("--source", default=str(SOURCE_DIR), help=f"Source spec root (default: {SOURCE_DIR})")
    parser.add_argument("--out", default=str(OUT_DIR), help=f"Output root (default: {OUT_DIR})")
    args = parser.parse_args()

    source_root = Path(args.source)
    out_root = Path(args.out)
    if not source_root.exists():
        print(f"No specs found at {source_root}", file=sys.stderr)
        sys.exit(1)

    spec_dirs = sorted(d for d in source_root.iterdir() if d.is_dir() and (d / "spec.json").exists())
    if args.spec:
        spec_dirs = [d for d in spec_dirs if d.name == args.spec]
        if not spec_dirs:
            raise SystemExit(f"No spec dir matching {args.spec!r}")

    corrupters = CORRUPTERS
    if args.mode:
        corrupters = [c for c in CORRUPTERS if c.id == args.mode]
        if not corrupters:
            raise SystemExit(f"No corrupter with id={args.mode!r}")

    applied: dict[str, int] = defaultdict(int)
    skipped: dict[str, int] = defaultdict(int)

    for spec_dir in spec_dirs:
        spec_dict = json.loads((spec_dir / "spec.json").read_text(encoding="utf-8"))
        for corrupter in corrupters:
            corruption = corrupter.apply(spec_dict)
            if corruption is None:
                skipped[corrupter.id] += 1
                continue
            target_dir = out_root / spec_dir.name / corrupter.id
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "spec.json").write_text(
                json.dumps(corruption.spec, indent=2), encoding="utf-8"
            )
            (target_dir / "corruption.json").write_text(
                json.dumps(
                    {
                        "description": corruption.description,
                        "changed_node_indices": corruption.changed_node_indices,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            applied[corrupter.id] += 1

    total_specs = len(spec_dirs)
    print()
    print(f"{'mode':<26}  applied / scanned")
    print("-" * 50)
    for c in corrupters:
        a = applied[c.id]
        s = skipped[c.id]
        print(f"{c.id:<26}  {a:>3} / {a + s:>3}  ({a / max(1, a + s):.0%})")
    print()
    print(f"Wrote corruptions across {total_specs} source specs into {out_root}")


if __name__ == "__main__":
    main()
