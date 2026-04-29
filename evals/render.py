"""Render every generated spec.json to render.png via lib.render.render_image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lib.render import render_image
from lib.types import Spec

DATA_DIR = Path(__file__).parent / "data"


def render_one(spec_path: Path, *, force: bool = False) -> bool:
    """Render spec.json next to it. Returns True if a render was produced."""
    out_path = spec_path.with_name("render.png")
    if out_path.exists() and not force:
        return False
    spec_dict = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = Spec(**spec_dict)
    render_image(
        spec,
        out_path,
        canvas_width=spec.canvas_width,
        canvas_height=spec.canvas_height,
        asset_dir=spec_path.parent,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Render generated specs")
    parser.add_argument("--mode", help="Render only this mode id")
    parser.add_argument("--force", action="store_true", help="Overwrite existing renders")
    parser.add_argument("--root", default=str(DATA_DIR), help=f"Data root (default: {DATA_DIR})")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"No data found at {root}; run `python -m evals.generate` first.", file=sys.stderr)
        sys.exit(1)

    spec_paths = sorted(root.glob("*/*/*/spec.json"))
    if args.mode:
        spec_paths = [p for p in spec_paths if p.parts[-4] == args.mode]

    rendered = skipped = 0
    for spec_path in spec_paths:
        rel = spec_path.relative_to(root)
        if render_one(spec_path, force=args.force):
            print(f"  rendered {rel}")
            rendered += 1
        else:
            skipped += 1
    print(f"Rendered {rendered}, skipped {skipped} (already present)")


if __name__ == "__main__":
    main()
