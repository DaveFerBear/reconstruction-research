"""Zero-shot agent that uses LLM to edit designs without preview images."""

import os
import json
from pathlib import Path
import litellm
from dotenv import load_dotenv

from .base import Agent
from lib.types import Spec

load_dotenv()


class ZeroShotAgent(Agent):
    """
    Zero-shot design editing agent.

    Uses an LLM (Gemini) to edit design specs based on text instructions,
    without providing a visual preview of the design.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-pro",
        temperature: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize the zero-shot agent.

        Args:
            model: LiteLLM model identifier (default: gemini/gemini-2.5-pro)
            temperature: Sampling temperature for generation (0.0-1.0)
            verbose: If True, print debug information
        """
        super().__init__(verbose=verbose)
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

    def edit(self, spec_path: Path, instruction: str, output_path: Path) -> Path:
        """
        Edit a design spec using zero-shot LLM prompting.

        Args:
            spec_path: Path to the input design spec (spec.json)
            instruction: Natural language instruction for the edit
            output_path: Path where the edited spec should be saved

        Returns:
            Path: The output path where the edited spec was saved
        """
        self.log(f"Loading spec from {spec_path}")
        spec = self.load_spec(spec_path)

        self.log(f"Applying instruction: '{instruction}'")
        edited_spec = self._apply_edit(spec, instruction)

        self.log(f"Saving edited spec to {output_path}")
        return self.save_spec(edited_spec, output_path)

    def _apply_edit(self, spec: Spec, instruction: str) -> Spec:
        """
        Apply an edit instruction to a spec using the LLM.

        Args:
            spec: The design specification
            instruction: Natural language instruction

        Returns:
            Spec: Modified design specification
        """
        # Convert spec to JSON string
        spec_json = json.dumps(spec.model_dump(), indent=2)

        # Build the prompt
        prompt = self._build_prompt(spec_json, instruction)

        self.log("Calling LLM...")

        # Call the LLM
        response = litellm.completion(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            api_key=self.api_key,
            temperature=self.temperature
        )

        # Extract the response
        response_text = response.choices[0].message.content.strip()

        self.log("Parsing LLM response...")

        # Parse the JSON response
        edited_spec_dict = self._extract_json(response_text)

        # Convert back to Spec object
        return Spec(**edited_spec_dict)

    def _build_prompt(self, spec_json: str, instruction: str) -> str:
        """
        Build the prompt for the LLM.

        Args:
            spec_json: JSON string of the design spec
            instruction: Edit instruction

        Returns:
            str: Formatted prompt
        """
        return f"""You are a design editing assistant. You will be given a design specification in JSON format and an instruction to modify it.

The design spec contains:
- canvas_width, canvas_height: Canvas dimensions
- background_color: Background color (hex format)
- nodes: Array of design elements, each with:
  - type: "text" or "image"
  - x, y: Position coordinates
  - width, height: Dimensions
  - rotation: Rotation in degrees
  - opacity: Opacity (0-1)

  For text nodes:
  - text: The text content
  - font-family, font-size, color, text-align, font-weight, font-style, text-decoration, text-transform

  For image nodes:
  - asset_description: Description of the image content

Your task:
1. Read the design spec carefully
2. Apply the requested modification to the spec
3. Return ONLY the modified JSON spec, with no additional text or explanation

Current design spec:
{spec_json}

Instruction: {instruction}

Return the modified design spec as valid JSON:"""

    def _extract_json(self, response_text: str) -> dict:
        """
        Extract JSON from LLM response.

        Args:
            response_text: Raw response from LLM

        Returns:
            dict: Parsed JSON object
        """
        # Try to find JSON in code blocks
        if "```json" in response_text:
            # Extract from markdown code block
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            # Extract from plain code block
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        else:
            # Assume entire response is JSON
            json_str = response_text.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}\n\nResponse:\n{response_text}")
