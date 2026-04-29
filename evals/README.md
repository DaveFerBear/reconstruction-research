# evals/

A benchmark for measuring whether VLMs can detect 9 common visual-design
failure modes:

1. Uneven distribution
2. Misalignment
3. Inconsistency (fonts, colors, weights)
4. Nonsensical scale/hierarchy
5. Crowding
6. Semiotic mismatch (icon ≠ label meaning) — **synthetic only**
7. Undesired overflow
8. Poor contrast
9. Undesired overlap

The eval has **two parallel data sources**:

- **synthetic** — hand-built specs (controlled experiment baseline)
- **real** — real Canva designs from `datasets/specs/`, programmatically
  corrupted to inject one isolated failure each (ecological validity)

`semiotic_mismatch` exists only in the synthetic eval; most real specs lack
swappable icon-label pairs.

## What this is *not*

This is **not** a harness for agents that fix or iterate on bad designs. There
is no repair loop. The pipeline ends at "VLM was asked → verdict recorded →
metric reported." The expected outcome is *low* judge accuracy on most modes
— that is the demonstration the benchmark exists to produce.

## Layout

```
evals/
├── common.py                       # specs/SVG helpers, MODE_DEFINITIONS, bbox + color utils
├── failure_modes/                  # SYNTHETIC: hand-built bad/good pairs
│   ├── base.py                     # FailureMode + Variant
│   └── <nine modules>.py
├── corrupters/                     # REAL: mutators over real spec.json files
│   ├── base.py                     # Corrupter + Corruption
│   └── <eight modules>.py          # (no semiotic_mismatch — see above)
├── generate.py                     # synthetic specs → data/
├── render.py                       # synthetic data/ → render.png
├── generate_real.py                # corrupt datasets/specs/ → data_real/
├── render_real.py                  # data_real/ → render.png (assets read from datasets/specs/)
├── judge.py                        # Anthropic SDK vision call → YES/NO
├── run_eval.py                     # orchestrator + metrics; --source {synthetic,real,both}
├── data/                           # gitignored — synthetic specs + renders
└── data_real/                      # gitignored — corrupted specs + renders
```

Specs use the existing `lib.types.Spec` schema (no new format).

## Usage — synthetic eval

```bash
python -m evals.generate          # 9 modes × 5 (bad,good) variants = 90 specs
python -m evals.render            # ~90 PNGs

open evals/data/poor_contrast/01/bad/render.png  # eyeball
```

## Usage — real-design corruption eval

The good case is the un-modified `datasets/specs/<id>/render.png`; the bad
case is a programmatic single-mutation of the spec rendered against the same
asset directory.

```bash
python -m evals.generate_real     # apply 8 corrupters across all real specs
python -m evals.render_real       # render every corrupted spec.json

# Eyeball one mode for one spec
open evals/data_real/<spec_id>/overflow/render.png datasets/specs/<spec_id>/render.png
```

`generate_real` prints a per-mode applicability summary — corrupters skip
specs that lack the structure to corrupt cleanly (e.g. a spec with one
TextNode is skipped by the `overlap` corrupter).

### Eyeball the corruptions in a browser

```bash
python -m evals.build_review     # writes evals/review.html
python -m http.server            # from the repo root
# open http://localhost:8000/evals/review.html
```

Side-by-side original vs corrupted with the corruption description, filter
by mode, rate good/bad/skip with `1`/`2`/`0` (persisted in localStorage),
navigate with `j`/`k`. "Export bad list" downloads a JSON of corruptions
flagged as bad — useful for diagnosing which corrupters need tuning.

## Run the judge (requires `ANTHROPIC_API_KEY`)

`ANTHROPIC_API_KEY` is read from the project-root `.env` via `python-dotenv`.

```bash
# Both sources, both models, all modes
python -m evals.run_eval

# One source
python -m evals.run_eval --source real
python -m evals.run_eval --source synthetic

# Scope further
python -m evals.run_eval --source real --mode overflow
python -m evals.run_eval --source real --limit 5         # 5 specs/mode for smoke
python -m evals.run_eval --model claude-sonnet-4-6
```

Results land in `evals/results.json` keyed by `<source>/<model>/<mode>` with
TP/TN/FP/FN, accuracy, precision, recall, and F1. Tables are printed once per
source.

## Notes

- **Prompt caching** is wired in (`cache_control` on the system prompt) for
  hygiene, but the per-mode system prompts are well below the 4096-token
  Opus minimum, so they will silently no-op. The dominant cost is per-image
  vision tokens, which are unique per call and never cacheable.
- **No thinking.** The judge omits `thinking` to measure visual perception
  rather than reasoning.
- **Volume.** Synthetic full run: ~90 renders × 2 models = ~180 calls. Real
  full run: ~300–500 corruptions × 2 (bad+good) × 2 models. Use `--limit` for
  smoke testing.
