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

## Run the judge

The judge is **open-ended**: each VLM is asked "what are the THREE biggest
design issues in this image?" with no taxonomy primer. Issue strings are
free-text (JSON array). A separate Haiku-backed classifier
(`evals/classify.py`) maps each issue back onto our 11-mode taxonomy after
the fact, so scoring is robust to vocabulary differences across providers.

Three models are evaluated by default: `claude-opus-4-6`, `claude-sonnet-4-6`,
`gpt-4o`. Both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are read from the
project-root `.env`.

### Two granularities of scoring

Modes roll up into two **supercategories**:

- **layout** (positional / spatial / sizing): uneven_distribution,
  misalignment, nonsensical_hierarchy, crowding, overflow, overlap
- **visual** (typography / color / iconography): inconsistency,
  semiotic_mismatch, poor_contrast, poor_colors, poor_fonts

For each (model, mode) we report **recall@3** (% of corrupted-for-this-mode
renders where the VLM flagged this mode in its top-3) and **FP rate** (% of
known-clean originals where the VLM flagged this mode anyway). Same metrics
at the supercategory level, plus per-model aggregates.

```bash
# Both sources, all 3 models
python -m evals.run_eval

# One source / one model
python -m evals.run_eval --source real --model gpt-4o

# Smoke test (5 variants/specs per mode)
python -m evals.run_eval --source real --limit 5

# Re-render the tables from existing results.json without re-paying judging
# (good for iterating on the scoring/display logic):
python -m evals.run_eval --reuse-judgments

# Skip the classifier entirely; uncached issues map to (None, None):
python -m evals.run_eval --skip-classifier
```

Results land in `evals/results.json`:

- `judgments[]` — every (render, model) judgment with raw VLM output, parsed
  issues, and post-classification mode/supercategory tags
- `classifications{}` — every unique free-text issue string mapped to
  `{mode, supercategory}`
- `summary` — confusion matrices + recall@3 + FP rate per (source, model,
  mode) and per (source, model, supercategory), plus per-model aggregates

The classifier persists to `evals/classifier_cache.json` (SHA256-keyed) so
re-runs that add new (model, render) pairs cost nearly zero classifier API
spend.

### Volume + cost

Full run is ~3 models × ~900 renders ≈ 2,700 vision-API calls + ~5–7K
classifier (Haiku) calls. Concurrency 8 → ~10–15 minutes wall clock and
~$30–40 in API spend (vision dominates). Use `--limit` for cheap smoke tests.
