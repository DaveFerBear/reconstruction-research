#!/usr/bin/env python3
"""Test script to verify the render functionality."""

import json
import sys
from pathlib import Path
from lib import render_image, Spec
from lib.ai import kontext_edit
from lib.utils import _to_data_url
import requests

specs_dir = Path('datasets/canva_specs')
output_dir = Path('datasets/reconstructions')
output_dir.mkdir(parents=True, exist_ok=True)

gen_images = '--gen-images' in sys.argv

def generate_assets(spec_data: dict, source_image_path: Path, output_dir: Path):
    """Generate image assets from the spec using kontext model."""
    nodes = spec_data.get('nodes', [])
    image_nodes = [n for n in nodes if n.get('type') == 'image']

    if not image_nodes:
        print("  No image nodes to generate")
        return

    # Convert source image to data URL
    print(f"  Converting source image from {source_image_path.name}...")
    source_url = _to_data_url(source_image_path)
    print(f"  ✓ Ready (data URL)")

    for idx, node in enumerate(image_nodes, start=1):
        description = node.get('asset_description', '')
        if not description:
            continue

        # Escape quotes and special characters in description
        description = description.replace('"', "'")

        print(f"  Generating asset-{idx}: {description[:60]}...")

        try:
            prompt = f"Extract the following image from this design: {description}"
            result = kontext_edit(prompt, source_url, with_logs=False)

            # Download the generated image
            if 'images' in result and result['images']:
                image_url = result['images'][0]['url']
                asset_path = output_dir / f"asset-{idx}.png"

                img_response = requests.get(image_url)
                img_response.raise_for_status()
                asset_path.write_bytes(img_response.content)
                print(f"  ✓ Saved {asset_path.name}")
            else:
                print(f"  ✗ No image returned for asset-{idx}")

        except Exception as e:
            print(f"  ✗ Error generating asset-{idx}: {e}")

if '--all' in sys.argv:
    # Render all specs
    for spec_path in specs_dir.glob('*/spec.json'):
        spec_data = json.load(spec_path.open())
        design_name = spec_path.parent.name
        design_output_dir = output_dir / design_name
        design_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = design_output_dir / "render.png"
        print(f"Rendering {design_name}...")

        # Generate assets if flag is set
        if gen_images:
            # Find the original source image
            source_images = list(Path('datasets/canva').glob(f'**/{design_name}.*'))
            if source_images:
                generate_assets(spec_data, source_images[0], design_output_dir)
            else:
                print(f"  Warning: No source image found for {design_name}")

        render_image(spec_data, output_path,
                    canvas_width=spec_data.get('canvas_width', 800),
                    canvas_height=spec_data.get('canvas_height', 600),
                    asset_dir=design_output_dir if gen_images else None)
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
