#!/usr/bin/env python3
"""Test script to verify the render functionality."""

import json
import sys
import asyncio
import os
from pathlib import Path
from lib import render_image, Spec
from lib.utils import _to_data_url
import aiohttp
from dotenv import load_dotenv

load_dotenv()

specs_dir = Path('datasets/canva_specs')
output_dir = Path('datasets/reconstructions')
output_dir.mkdir(parents=True, exist_ok=True)

gen_images = '--gen-images' in sys.argv


async def kontext_edit_async(prompt: str, image_url: str, session: aiohttp.ClientSession) -> dict:
    """Async version of kontext_edit."""
    import os
    fal_api_key = os.getenv("FAL_API_KEY")

    headers = {"Authorization": f"Key {fal_api_key}"}
    payload = {"prompt": prompt, "image_url": image_url}

    async with session.post("https://fal.run/fal-ai/flux-pro/kontext", json=payload, headers=headers) as response:
        if response.status == 422:
            error_text = await response.text()
            raise Exception(f"API Error: {error_text}")
        response.raise_for_status()

        if response.status == 200:
            return await response.json()
        elif response.status == 202:
            job = await response.json()
            status_url = job.get("status_url") or job.get("response_url")

            # Poll for completion
            while True:
                await asyncio.sleep(2)
                async with session.get(status_url, headers=headers) as status_response:
                    status_response.raise_for_status()
                    data = await status_response.json()
                    state = (data.get("status") or data.get("state") or "").lower()

                    if state in ("completed", "success", "succeeded"):
                        return data
                    if state in ("failed", "error"):
                        raise RuntimeError(f"FAL job failed: {data}")


async def download_image(url: str, path: Path, session: aiohttp.ClientSession):
    """Download image to path."""
    async with session.get(url) as response:
        response.raise_for_status()
        content = await response.read()
        path.write_bytes(content)


async def generate_single_asset(idx: int, description: str, source_url: str, output_path: Path, session: aiohttp.ClientSession):
    """Generate a single asset."""
    description = description.replace('"', "'")
    print(f"  Generating asset-{idx}: {description[:60]}...")

    try:
        prompt = f"Extract the following image from this design: {description}"
        result = await kontext_edit_async(prompt, source_url, session)

        if 'images' in result and result['images']:
            image_url = result['images'][0]['url']
            await download_image(image_url, output_path, session)
            print(f"  ✓ Saved {output_path.name}")
        else:
            print(f"  ✗ No image returned for asset-{idx}")
    except Exception as e:
        print(f"  ✗ Error generating asset-{idx}: {e}")


async def generate_assets(spec_data: dict, source_image_path: Path, output_dir: Path):
    """Generate image assets from the spec using kontext model (async)."""
    print(f"  Converting source image from {source_image_path.name}...")
    source_url = _to_data_url(source_image_path)
    print(f"  ✓ Ready (data URL)")

    async with aiohttp.ClientSession() as session:
        tasks = []

        # Generate background image if needed
        if spec_data.get('has_background_image') and spec_data.get('background_image_description'):
            bg_description = spec_data['background_image_description'].replace('"', "'")
            print(f"  Generating background: {bg_description[:60]}...")

            async def gen_background():
                try:
                    prompt = f"Extract the following image from this design: {bg_description}"
                    result = await kontext_edit_async(prompt, source_url, session)

                    if 'images' in result and result['images']:
                        bg_path = output_dir / "background.png"
                        await download_image(result['images'][0]['url'], bg_path, session)
                        print(f"  ✓ Saved background.png")
                    else:
                        print(f"  ✗ No image returned for background")
                except Exception as e:
                    print(f"  ✗ Error generating background: {e}")

            tasks.append(gen_background())

        # Generate node assets
        nodes = spec_data.get('nodes', [])
        image_nodes = [n for n in nodes if n.get('type') == 'image']

        for idx, node in enumerate(image_nodes, start=1):
            description = node.get('asset_description', '')
            if not description:
                continue

            asset_path = output_dir / f"asset-{idx}.png"
            tasks.append(generate_single_asset(idx, description, source_url, asset_path, session))

        # Run all generations in parallel
        if tasks:
            await asyncio.gather(*tasks)

async def generate_all_assets():
    """Generate assets for all designs (async)."""
    tasks = []
    for spec_path in specs_dir.glob('*/spec.json'):
        spec_data = json.load(spec_path.open())
        design_name = spec_path.parent.name
        design_output_dir = output_dir / design_name
        design_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {design_name}...")

        # Find the original source image
        source_images = list(Path('datasets/canva').glob(f'**/{design_name}.*'))
        if source_images:
            tasks.append(generate_assets(spec_data, source_images[0], design_output_dir))
        else:
            print(f"  Warning: No source image found for {design_name}")

    if tasks:
        await asyncio.gather(*tasks)


def render_all_designs():
    """Render all designs (sync, after assets are generated)."""
    for spec_path in specs_dir.glob('*/spec.json'):
        spec_data = json.load(spec_path.open())
        design_name = spec_path.parent.name
        design_output_dir = output_dir / design_name
        output_path = design_output_dir / "render.png"

        print(f"Rendering {design_name}...")
        render_image(spec_data, output_path,
                    canvas_width=spec_data.get('canvas_width', 800),
                    canvas_height=spec_data.get('canvas_height', 600),
                    asset_dir=design_output_dir if gen_images else None)


if __name__ == '__main__':
    if '--all' in sys.argv:
        if gen_images:
            # First generate all assets async
            asyncio.run(generate_all_assets())

        # Then render all designs sync
        render_all_designs()
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
