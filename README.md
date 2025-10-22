# Reconstruction Research

Research project for deconstructing and reconstructing graphic designs using AI.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

## Usage

```bash
# Render all designs
python test_render.py --all

# Render with asset generation
python test_render.py --all --gen-images
```

## Editor

Preview designs in your browser:

1. Start a local server:
```bash
python -m http.server 8000
```

2. Open in browser:
```
http://localhost:8000/editor/
```

Click any spec in the sidebar to preview it.
