#!/usr/bin/env python3
"""Test the agent framework with a simple design edit."""

import sys
from pathlib import Path
from agents import ZeroShotAgent

def main():
    # Example usage
    spec_path = Path("datasets/canva_specs/1600w-zcHQ3XLP3Ow/spec.json")
    output_path = Path("outputs/test_agent/spec.json")
    instruction = "Make the main title text blue and increase its font size by 20%"

    if not spec_path.exists():
        print(f"Error: Spec not found at {spec_path}")
        print("Please provide a valid spec path as the first argument")
        sys.exit(1)

    # Allow custom instruction and paths via command line
    if len(sys.argv) > 1:
        spec_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        instruction = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = Path(sys.argv[3])

    print("=" * 80)
    print("ZERO-SHOT AGENT TEST")
    print("=" * 80)
    print(f"Input spec:   {spec_path}")
    print(f"Instruction:  {instruction}")
    print(f"Output spec:  {output_path}")
    print()

    # Create agent and apply edit
    agent = ZeroShotAgent(verbose=True)

    try:
        result_path = agent.edit(
            spec_path=spec_path,
            instruction=instruction,
            output_path=output_path
        )

        print()
        print("=" * 80)
        print(f"✓ Successfully saved edited spec to: {result_path}")
        print()
        print("To render the edited design:")
        print(f"  python test_render.py {result_path.parent.name}")
    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
