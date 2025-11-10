#!/usr/bin/env python3
"""Caption all images in specs using GPT-5 vision API."""

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from lib.caption import caption_image

SPECS_DIR = Path("datasets/specs")


def process_spec(spec_path: Path) -> dict:
    """Process a single spec: caption all images and update descriptions."""
    design_name = spec_path.parent.name

    try:
        # Load spec
        with open(spec_path, 'r') as f:
            spec_data = json.load(f)

        updated = False
        captions_generated = 0

        # Caption background image if it exists
        if spec_data.get('has_background_image'):
            bg_path = spec_path.parent / 'background.png'
            if bg_path.exists():
                print(f"  [{design_name}] Captioning background image...")
                caption = caption_image(bg_path)
                if caption:
                    spec_data['background_image_description'] = caption
                    updated = True
                    captions_generated += 1

        # Caption all image nodes
        for node in spec_data.get('nodes', []):
            if node.get('type') == 'image' and node.get('filename'):
                image_path = spec_path.parent / node['filename']
                if image_path.exists():
                    print(f"  [{design_name}] Captioning {node['filename']}...")
                    caption = caption_image(image_path)
                    if caption:
                        node['asset_description'] = caption
                        updated = True
                        captions_generated += 1

        # Save updated spec
        if updated:
            with open(spec_path, 'w') as f:
                json.dump(spec_data, f, indent=2, ensure_ascii=False)
            print(f" [{design_name}] Updated {captions_generated} captions")
        else:
            print(f"  [{design_name}] No images to caption")

        return {
            'design': design_name,
            'success': True,
            'captions': captions_generated
        }

    except Exception as e:
        print(f" [{design_name}] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'design': design_name,
            'success': False,
            'error': str(e)
        }


def main():
    """Caption all images in all specs."""
    # Find all specs
    spec_paths = sorted(SPECS_DIR.glob('*/spec.json'))

    if not spec_paths:
        print("No specs found!")
        return

    print(f"Found {len(spec_paths)} specs to process")
    print(f"Using GPT-5 for image captioning\n")

    # Process in parallel (max 5 concurrent to avoid rate limits)
    max_workers = 5
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_spec, spec_path): spec_path for spec_path in spec_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing specs"):
            result = future.result()
            results.append(result)

    # Print summary
    successful = sum(1 for r in results if r['success'])
    total_captions = sum(r.get('captions', 0) for r in results if r['success'])

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Specs processed:     {successful}/{len(spec_paths)}")
    print(f"Total captions:      {total_captions}")
    print(f"Average per spec:    {total_captions/successful if successful > 0 else 0:.1f}")

    # Show failed specs
    failed = [r for r in results if not r['success']]
    if failed:
        print(f"\nFailed specs:")
        for r in failed:
            print(f"  - {r['design']}: {r.get('error', 'Unknown error')}")


if __name__ == '__main__':
    main()
