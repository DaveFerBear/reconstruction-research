#!/usr/bin/env python3
"""Test script to verify the render functionality."""

import json
import sys
from pathlib import Path
from lib import render_image, Spec

specs_dir = Path('datasets/canva_specs')
output_dir = Path('datasets/reconstructions')
output_dir.mkdir(parents=True, exist_ok=True)

if '--all' in sys.argv:
    # Render all specs
    for spec_path in specs_dir.glob('*/spec.json'):
        spec_data = json.load(spec_path.open())
        design_name = spec_path.parent.name
        design_output_dir = output_dir / design_name
        design_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = design_output_dir / "render.png"
        print(f"Rendering {design_name}...")
        render_image(spec_data, output_path,
                    canvas_width=spec_data.get('canvas_width', 800),
                    canvas_height=spec_data.get('canvas_height', 600))
else:
    # Render single test
    spec_path = Path('datasets/canva_specs/1600w-1HZYAUid2AE/spec.json')
    output_path = Path('datasets/reconstructions/test_render.png')

    print(f"Loading spec from: {spec_path}")
    spec_data = json.load(spec_path.open())

    print(f"Rendering to: {output_path}")
    result = render_image(spec_data, output_path, canvas_width=800, canvas_height=600)

    print(f"Successfully rendered to: {result}")
    print(f"File exists: {result.exists()}")
    print(f"File size: {result.stat().st_size} bytes")
