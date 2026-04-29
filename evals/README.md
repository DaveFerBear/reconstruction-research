# evals/

A benchmark for measuring whether VLMs can detect 9 common visual-design
failure modes:

1. Uneven distribution
2. Misalignment
3. Inconsistency (fonts, colors, weights)
4. Nonsensical scale/hierarchy
5. Crowding
6. Semiotic mismatch (icon ≠ label meaning)
7. Undesired overflow
8. Poor contrast
9. Undesired overlap

## What this is *not*

This is **not** a harness for agents that fix or iterate on bad designs. There
is no repair loop. The pipeline ends at "VLM was asked → verdict recorded →
metric reported." The expected outcome is *low* judge accuracy on most modes
— that is the demonstration the benchmark exists to produce.

## Layout

```
evals/
├── common.py                       # spec / SVG / icon helpers
├── failure_modes/
│   ├── base.py                     # FailureMode + Variant dataclasses
│   ├── __init__.py                 # FAILURE_MODES registry
│   └── <nine modules>.py
├── generate.py                     # spec.json + svg sidecars → data/
├── render.py                       # data/.../spec.json → render.png via lib.render
├── judge.py                        # Anthropic SDK vision call → YES/NO
├── run_eval.py                     # orchestrator + metrics
└── data/                           # generated, gitignored
    └── <mode>/<NN>/<bad|good>/{spec.json, render.png, svg-*.svg}
```

Specs use the existing `lib.types.Spec` schema (no new format). Each failure
mode emits 5 hand-tuned `(bad, good)` variant pairs.

## Usage

### Generate specs and renders (no API key required)

```bash
python -m evals.generate
python -m evals.render
```

Inspect a few outputs visually:

```bash
open evals/data/poor_contrast/01/bad/render.png
open evals/data/poor_contrast/01/good/render.png
```

### Run the judge (requires `ANTHROPIC_API_KEY`)

`ANTHROPIC_API_KEY` is read from the project-root `.env` via `python-dotenv`.

```bash
# Both models (claude-opus-4-6 + claude-sonnet-4-6), all modes
python -m evals.run_eval

# One mode
python -m evals.run_eval --mode overflow

# One model
python -m evals.run_eval --model claude-sonnet-4-6

# Smoke test (2 variants per mode)
python -m evals.run_eval --limit 2
```

Results land in `evals/data/results.json` keyed by `<model>/<mode>` with
TP/TN/FP/FN, accuracy, precision, recall, and F1.

## Notes

- **Prompt caching** is wired in (`cache_control` on the system prompt) for
  hygiene, but the per-mode system prompts are well below the 4096-token
  Opus minimum, so they will silently no-op (`cache_creation_input_tokens`
  stays at 0). The dominant cost is per-image vision tokens, which are
  unique per call and never cacheable. Don't expect cache hits.
- **No thinking.** The judge omits `thinking` to measure visual perception
  rather than reasoning. Adding thinking would let the model reason about
  which failure is statistically likely instead of looking at the image —
  muddying the "VLM blindness" signal.
- **Volume.** A full run is ~90 renders × 2 models = ~180 API calls. At
  roughly 2k input + 50 output tokens per call, this is comfortably under
  any rate-limit or cost concern.
