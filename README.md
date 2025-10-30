# Reconstruction Research

Research project for deconstructing and reconstructing graphic designs using AI.

## Repository Structure

```
├── lib/                    # Core library modules
│   ├── ai.py              # AI model integrations (FAL, Gemini)
│   ├── render.py          # HTML/CSS rendering engine
│   ├── types.py           # Pydantic models for design specs
│   ├── uiclip.py          # UIClip aesthetic scoring model
│   ├── aesexpert.py       # HumanAesExpert scoring (experimental)
│   ├── utils.py           # Utility functions
│   └── prompts.py         # LLM prompts for evaluation
│
├── datasets/              # Design datasets
│   ├── canva/            # Original Canva design images
│   ├── canva_specs/      # JSON specifications for each design
│   ├── reconstructions/  # Rendered reconstructions + assets
│   └── aesthetic_scores/ # Evaluation results (JSON)
│
├── editor/               # Web-based design editor
│   └── index.html       # Interactive preview and editing UI
│
├── notebooks/           # Jupyter notebooks for analysis
│   ├── explore.ipynb   # Aesthetic evaluation & visualization
│   └── deconstruct.ipynb
│
└── test_*.py           # Test/evaluation scripts
    ├── test_render.py        # Batch rendering
    ├── test_edit.py          # Image editing tests
    └── test_uiclip_eval.py   # UIClip scoring
```

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

Set up environment variables in `.env`:
```
FAL_API_KEY=your_fal_key
GEMINI_API_KEY=your_gemini_key
```

## Usage

### Rendering Designs

```bash
# Render all designs
python test_render.py --all

# Render with asset generation
python test_render.py --all --gen-images
```

### Aesthetic Evaluation

```bash
# Score all designs with UIClip
python test_uiclip_eval.py
```

Or use the notebooks for interactive analysis:
```bash
jupyter notebook notebooks/explore.ipynb
```

### Editor

Preview and edit designs in your browser:

1. Start a local server:
```bash
python -m http.server 8000
```

2. Open in browser:
```
http://localhost:8000/editor/
```

Click any spec in the sidebar to preview it. Edit properties like text, colors, positions, and opacity in real-time.

## Key Components

### Design Specification Format

Each design is represented as a JSON spec with:
- **Canvas properties**: dimensions, background color
- **Nodes**: text and image elements with positioning, styling, and opacity
- **Assets**: extracted images stored separately

See `datasets/canva_specs/` for examples.

### Rendering Pipeline

1. **Spec parsing** (`lib/types.py`) - Pydantic models validate design specs
2. **HTML generation** (`lib/render.py`) - Converts specs to positioned HTML/CSS
3. **Screenshot** - Playwright captures the rendered design
4. **Asset handling** (`lib/ai.py`) - Optional AI-based image generation/editing

### Aesthetic Evaluation

Multiple models for scoring design quality:
- **UIClip** - Zero-shot classification (0-1 score)
- **Gemini 2.5 Pro** - Vision LLM scoring (0-100 scale)

Results compare original Canva designs vs. reconstructed versions.
