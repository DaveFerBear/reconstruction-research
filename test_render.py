#!/usr/bin/env python3
"""Test script to verify the render functionality."""

import json
from pathlib import Path
from lib import render_image, Spec

# Load one of the existing specs
spec_path = Path('datasets/canva_specs/1600w-1HZYAUid2AE/spec.json')
output_path = Path('datasets/reconstructions/test_render.png')

print(f"Loading spec from: {spec_path}")
with open(spec_path, 'r') as f:
    spec_data = json.load(f)

print(f"Rendering to: {output_path}")
result = render_image(spec_data, output_path, canvas_width=800, canvas_height=600)

print(f"Successfully rendered to: {result}")
print(f"File exists: {result.exists()}")
print(f"File size: {result.stat().st_size} bytes")
