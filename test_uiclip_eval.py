#!/usr/bin/env python3
"""Evaluate UI quality scores for all Canva images and reconstructions."""

from pathlib import Path
from PIL import Image
from lib.uiclip import score_image


def main():
    canva_dir = Path('datasets/canva')
    reconstructions_dir = Path('datasets/reconstructions')

    print("Evaluating UI quality scores...")
    print("=" * 80)

    # Get all original Canva images
    canva_images = sorted(canva_dir.glob('*.webp'))

    results = []

    for img_path in canva_images:
        design_name = img_path.stem

        print(f"\n{design_name}:")
        print("-" * 40)

        # Score original
        print(f"  Scoring original...")
        original_score = score_image(img_path, description="graphic design")
        print(f"  Original:        {original_score:.4f}")

        # Score reconstruction if it exists
        reconstruction_path = reconstructions_dir / design_name / "render.png"
        if reconstruction_path.exists():
            print(f"  Scoring reconstruction...")
            reconstruction_score = score_image(reconstruction_path, description="graphic design")
            print(f"  Reconstruction:  {reconstruction_score:.4f}")
            diff = reconstruction_score - original_score
            print(f"  Difference:      {diff:+.4f}")

            results.append({
                'name': design_name,
                'original': original_score,
                'reconstruction': reconstruction_score,
                'diff': diff
            })
        else:
            print(f"  Reconstruction:  (not found)")
            results.append({
                'name': design_name,
                'original': original_score,
                'reconstruction': None,
                'diff': None
            })

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        # Filter results with reconstructions
        results_with_recon = [r for r in results if r['reconstruction'] is not None]

        if results_with_recon:
            avg_original = sum(r['original'] for r in results_with_recon) / len(results_with_recon)
            avg_reconstruction = sum(r['reconstruction'] for r in results_with_recon) / len(results_with_recon)
            avg_diff = sum(r['diff'] for r in results_with_recon) / len(results_with_recon)

            print(f"\nAverage Original Score:        {avg_original:.4f}")
            print(f"Average Reconstruction Score:  {avg_reconstruction:.4f}")
            print(f"Average Difference:            {avg_diff:+.4f}")

            # Find best and worst
            best = max(results_with_recon, key=lambda r: r['reconstruction'])
            worst = min(results_with_recon, key=lambda r: r['reconstruction'])

            print(f"\nBest Reconstruction:  {best['name']} ({best['reconstruction']:.4f})")
            print(f"Worst Reconstruction: {worst['name']} ({worst['reconstruction']:.4f})")


if __name__ == '__main__':
    main()
