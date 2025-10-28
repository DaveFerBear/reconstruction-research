#!/usr/bin/env python3
"""Evaluate aesthetic scores for all Canva images and reconstructions using HumanAesExpert."""

import os
# Force PyTorch backend before any imports
os.environ["TRANSFORMERS_BACKEND"] = "pytorch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

from pathlib import Path
from PIL import Image
from lib.aesexpert import score_image, get_expert_scores
import json


def main():
    canva_dir = Path('datasets/canva')
    reconstructions_dir = Path('datasets/reconstructions')
    output_dir = Path('datasets/aesthetic_scores')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating aesthetic scores with HumanAesExpert...")
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
        original_score = score_image(img_path, question="Rate the aesthetics of this graphic design.")
        print(f"  Original score:        {original_score:.4f}")

        # Get expert scores for original
        print(f"  Getting expert scores...")
        original_expert_scores, original_expert_texts = get_expert_scores(img_path)

        # Score reconstruction if it exists
        reconstruction_path = reconstructions_dir / design_name / "render.png"
        if reconstruction_path.exists():
            print(f"  Scoring reconstruction...")
            reconstruction_score = score_image(reconstruction_path, question="Rate the aesthetics of this graphic design.")
            print(f"  Reconstruction score:  {reconstruction_score:.4f}")
            diff = reconstruction_score - original_score
            print(f"  Difference:            {diff:+.4f}")

            # Get expert scores for reconstruction
            reconstruction_expert_scores, reconstruction_expert_texts = get_expert_scores(reconstruction_path)

            results.append({
                'name': design_name,
                'original_score': original_score,
                'reconstruction_score': reconstruction_score,
                'diff': diff,
                'original_expert_scores': original_expert_scores,
                'original_expert_texts': original_expert_texts,
                'reconstruction_expert_scores': reconstruction_expert_scores,
                'reconstruction_expert_texts': reconstruction_expert_texts,
            })
        else:
            print(f"  Reconstruction:  (not found)")
            results.append({
                'name': design_name,
                'original_score': original_score,
                'reconstruction_score': None,
                'diff': None,
                'original_expert_scores': original_expert_scores,
                'original_expert_texts': original_expert_texts,
                'reconstruction_expert_scores': None,
                'reconstruction_expert_texts': None,
            })

        # Save individual result
        result_file = output_dir / f"{design_name}.json"
        with result_file.open('w') as f:
            json.dump(results[-1], f, indent=2, default=str)

    # Save all results
    all_results_file = output_dir / "all_scores.json"
    with all_results_file.open('w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        results_with_recon = [r for r in results if r['reconstruction_score'] is not None]

        if results_with_recon:
            avg_original = sum(r['original_score'] for r in results_with_recon) / len(results_with_recon)
            avg_reconstruction = sum(r['reconstruction_score'] for r in results_with_recon) / len(results_with_recon)
            avg_diff = sum(r['diff'] for r in results_with_recon) / len(results_with_recon)

            print(f"\nAverage Original Score:        {avg_original:.4f}")
            print(f"Average Reconstruction Score:  {avg_reconstruction:.4f}")
            print(f"Average Difference:            {avg_diff:+.4f}")

            # Find best and worst
            best = max(results_with_recon, key=lambda r: r['reconstruction_score'])
            worst = min(results_with_recon, key=lambda r: r['reconstruction_score'])

            print(f"\nBest Reconstruction:  {best['name']} ({best['reconstruction_score']:.4f})")
            print(f"Worst Reconstruction: {worst['name']} ({worst['reconstruction_score']:.4f})")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - Individual scores: {output_dir}/<design_name>.json")
    print(f"  - All scores: {all_results_file}")


if __name__ == '__main__':
    main()
