#!/usr/bin/env python3
"""Test the agent framework with a simple design edit."""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from agents import ZeroShotAgent
from agents.singleshot import SingleShotAgent

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test agent-based design editing")
    parser.add_argument("spec_path", nargs="?", default="datasets/specs/1236w-JT1Z7rx2PuU/spec.json",
                        help="Path to input spec.json")
    parser.add_argument("instruction", nargs="?",
                        default="Make the main title text blue and increase its font size by 20%",
                        help="Edit instruction")
    parser.add_argument("output_path", nargs="?", default=None,
                        help="Path to save edited spec (default: auto-generate from design + agent + timestamp)")
    parser.add_argument("--agent", default="ZeroShotAgent",
                        choices=["ZeroShotAgent", "SingleShotAgent"],
                        help="Agent type to use (default: ZeroShotAgent)")

    args = parser.parse_args()

    spec_path = Path(args.spec_path)
    instruction = args.instruction

    # Auto-generate output path if not provided
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        # Extract design name from spec path
        design_name = spec_path.parent.name

        # Get agent ID (lowercase, simplified)
        agent_id = args.agent.lower().replace("agent", "")

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create edit ID: {design}_{agentid}_{timestamp}
        edit_id = f"{design_name}_{agent_id}_{timestamp}"
        output_path = Path(f"edits/{edit_id}/spec.json")

    if not spec_path.exists():
        print(f"Error: Spec not found at {spec_path}")
        print("Please provide a valid spec path as the first argument")
        sys.exit(1)

    print("=" * 80)
    print(f"{args.agent.upper().replace('AGENT', ' AGENT')} TEST")
    print("=" * 80)
    print(f"Agent:        {args.agent}")
    print(f"Edit ID:      {output_path.parent.name}")
    print(f"Input spec:   {spec_path}")
    print(f"Instruction:  {instruction}")
    print(f"Output spec:  {output_path}")
    print()

    # Create agent based on type
    if args.agent == "SingleShotAgent":
        agent = SingleShotAgent(verbose=True)
    else:
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

        # Check if render was created
        render_path = result_path.parent / "render.png"
        if render_path.exists():
            print(f"✓ Rendered design saved to: {render_path}")
        else:
            print()
            print("To render the edited design:")
            print(f"  python test_render.py {result_path.parent}")
    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
