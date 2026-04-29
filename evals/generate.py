"""Write spec.json + svg sidecars for every (failure_mode, variant, bad/good).

Output layout:
    evals/data/<mode_id>/<NN>/<bad|good>/spec.json
                                        /svg-*.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evals.common import write_spec_dir
from evals.failure_modes import FAILURE_MODES

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate failure-mode specs")
    parser.add_argument("--mode", help="Generate only this mode id")
    parser.add_argument(
        "--out",
        default=str(DATA_DIR),
        help=f"Output root (default: {DATA_DIR})",
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    modes = FAILURE_MODES
    if args.mode:
        modes = [m for m in FAILURE_MODES if m.id == args.mode]
        if not modes:
            raise SystemExit(f"No failure mode with id={args.mode!r}")

    written = 0
    for mode in modes:
        variants = mode.generate()
        print(f"[{mode.id}] {len(variants)} variants")
        for i, variant in enumerate(variants, start=1):
            variant_dir = out_root / mode.id / f"{i:02d}"
            write_spec_dir(variant_dir / "bad", variant.bad_spec, variant.bad_svgs)
            write_spec_dir(variant_dir / "good", variant.good_spec, variant.good_svgs)
            written += 2
    print(f"Wrote {written} specs to {out_root}")


if __name__ == "__main__":
    main()
