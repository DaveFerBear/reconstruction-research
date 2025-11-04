"""Direct image editing agent - theoretical upper bound for aesthetic quality."""

import os
import json
import tempfile
import requests
from pathlib import Path
from dotenv import load_dotenv

from .base import Agent
from lib.types import Spec
from lib.render import render_image
from lib.ai import kontext_edit
from lib.utils import _to_data_url

load_dotenv()


class DirectEditAgent(Agent):
    """
    Direct image editing agent.

    Establishes the theoretical upper bound for aesthetic quality by editing
    the rendered design as a single image (not constrained by JSON spec format).
    Returns a simple single-layer spec with the edited image as background.
    """

    def __init__(
        self,
        verbose: bool = False
    ):
        """
        Initialize the direct edit agent.

        Args:
            verbose: If True, print debug information
        """
        super().__init__(verbose=verbose)

    def edit(self, spec_path: Path, instruction: str, output_path: Path) -> Path:
        """
        Edit a design by rendering it and using image-to-image editing.

        Args:
            spec_path: Path to the input design spec (spec.json)
            instruction: Natural language instruction for the edit
            output_path: Path where the edited spec should be saved

        Returns:
            Path: The output path where the edited spec was saved
        """
        self.log(f"Loading spec from {spec_path}")
        spec = self.load_spec(spec_path)

        self.log(f"Applying instruction via direct image editing: '{instruction}'")

        # Determine asset directory
        asset_dir = spec_path.parent

        edited_spec = self._apply_edit(spec, instruction, asset_dir, output_path)

        self.log(f"Saving edited spec to {output_path}")
        return self.save_spec(edited_spec, output_path)

    def _apply_edit(self, spec: Spec, instruction: str, asset_dir: Path, output_path: Path) -> Spec:
        """
        Apply an edit by rendering the design and using an image editing model.

        Args:
            spec: The design specification
            instruction: Natural language instruction
            asset_dir: Directory containing asset images for rendering
            output_path: Output path for the spec (used to save edited image nearby)

        Returns:
            Spec: New specification with edited image as background
        """
        # Render the original design to a temporary image
        self.log("Rendering original design...")
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            render_image(
                spec=spec,
                output_path=tmp_path,
                canvas_width=spec.canvas_width,
                canvas_height=spec.canvas_height,
                asset_dir=asset_dir
            )

            # Convert to data URL (used throughout the codebase for FAL API)
            self.log("Converting image to data URL...")
            image_url = _to_data_url(tmp_path)

            self.log(f"Editing image with instruction: '{instruction}'")
            result = kontext_edit(
                prompt=instruction,
                image_url=image_url,
                with_logs=self.verbose
            )

            # Download the edited image
            edited_image_url = result['images'][0]['url']
            self.log(f"Downloading edited image from {edited_image_url}")

            # Save edited image next to the output spec
            output_dir = output_path.parent
            edited_image_path = output_dir / 'edited_image.png'

            response = requests.get(edited_image_url)
            response.raise_for_status()

            with open(edited_image_path, 'wb') as f:
                f.write(response.content)

            self.log(f"Saved edited image to {edited_image_path}")

        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()

        # Create a simple spec with just the edited image as background
        new_spec = Spec(
            canvas_width=spec.canvas_width,
            canvas_height=spec.canvas_height,
            background_color="#FFFFFF",
            has_background_image=True,
            background_image_description=f"Edited design: {instruction}",
            nodes=[]  # No nodes, just the background image
        )

        # Copy the edited image to 'background.png' so it gets picked up by rendering
        background_path = output_dir / 'background.png'
        if edited_image_path.exists():
            import shutil
            shutil.copy2(edited_image_path, background_path)
            self.log(f"Copied edited image to {background_path}")

        return new_spec

    def _upload_image_to_fal(self, image_path: Path) -> str:
        """
        Upload an image to FAL's storage and return the URL.

        Args:
            image_path: Path to the image file

        Returns:
            str: URL of the uploaded image
        """
        import requests

        FAL_API_KEY = os.getenv("FAL_API_KEY")
        if not FAL_API_KEY:
            raise ValueError("FAL_API_KEY environment variable not set")

        # FAL storage upload endpoint
        upload_url = "https://fal.run/fal-ai/files/upload"

        headers = {
            "Authorization": f"Key {FAL_API_KEY}",
        }

        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/png')}
            response = requests.post(upload_url, headers=headers, files=files)
            response.raise_for_status()

        result = response.json()
        return result['url']
