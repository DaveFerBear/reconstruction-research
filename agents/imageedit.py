"""ImageEdit agent that edits the entire composited design using nano-banana."""

import os
import tempfile
from pathlib import Path
import requests
from dotenv import load_dotenv

from .base import Agent
from lib.types import Spec
from lib.render import render_image
from lib.ai import edit_image
from lib.utils import _to_data_url

load_dotenv()


class ImageEditAgent(Agent):
    """
    Image edit agent that treats the design as a single composited image.

    This agent:
    1. Renders the entire spec (all layers) into a single image
    2. Uses nano-banana to edit that rendered image
    3. Returns a new spec with zero nodes and the edited image as background
    """

    def __init__(
        self,
        timeout: int = 120,
        verbose: bool = False
    ):
        """
        Initialize the image edit agent.

        Args:
            timeout: Request timeout in seconds for nano-banana API (default: 120)
            verbose: If True, print debug information
        """
        super().__init__(verbose=verbose)
        self.timeout = timeout

        # Load API key
        self.fal_api_key = os.getenv("FAL_API_KEY")
        if not self.fal_api_key:
            raise ValueError("FAL_API_KEY environment variable not set")

    def edit(self, spec_path: Path, instruction: str, output_path: Path) -> Path:
        """
        Edit a design by rendering it, editing the composited image, and saving as background.

        Args:
            spec_path: Path to the input design spec (spec.json)
            instruction: Natural language instruction for nano-banana
            output_path: Path where the edited spec should be saved

        Returns:
            Path: The output path where the edited spec was saved
        """
        self.log(f"Loading spec from {spec_path}")
        spec = self.load_spec(spec_path)

        # Set up output directory
        source_dir = spec_path.parent
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Render the original design to a temporary file
        self.log("Rendering design to composited image...")
        temp_render_fd, temp_render_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_render_fd)
        temp_render_path = Path(temp_render_path)

        try:
            render_image(
                spec=spec,
                output_path=temp_render_path,
                canvas_width=spec.canvas_width,
                canvas_height=spec.canvas_height,
                asset_dir=source_dir
            )
            self.log(f"✓ Rendered to temporary file")

            # Step 2: Convert rendered image to data URL for nano-banana
            self.log("Converting rendered image to data URL...")
            rendered_data_url = _to_data_url(temp_render_path)

            # Step 3: Call nano-banana to edit the image
            self.log(f"Editing composited image with nano-banana: '{instruction}'")
            result = edit_image(
                prompt=instruction,
                image_urls=[rendered_data_url],
                timeout=self.timeout
            )

            # Step 4: Extract the edited image URL from result
            if "images" in result and len(result["images"]) > 0:
                edited_image_url = result["images"][0]["url"]
            elif "output" in result and "images" in result["output"]:
                edited_image_url = result["output"]["images"][0]["url"]
            else:
                raise ValueError(f"Could not extract edited image from nano-banana result: {result}")

            # Step 5: Download and save the edited image as background.png
            background_path = output_dir / "background.png"
            self.log(f"Downloading edited image to {background_path}")

            if edited_image_url.startswith('http'):
                response = requests.get(edited_image_url, timeout=self.timeout)
                response.raise_for_status()
                with open(background_path, 'wb') as f:
                    f.write(response.content)
            elif edited_image_url.startswith('data:'):
                # Handle data URL
                import base64
                if ',' in edited_image_url:
                    header, encoded = edited_image_url.split(',', 1)
                    image_data = base64.b64decode(encoded)
                    with open(background_path, 'wb') as f:
                        f.write(image_data)
            else:
                raise ValueError(f"Unknown image URL format: {edited_image_url}")

            self.log(f"✓ Saved edited image to {background_path}")

            # Step 6: Create a new spec with the edited image as background and no nodes
            edited_spec = Spec(
                canvas_width=spec.canvas_width,
                canvas_height=spec.canvas_height,
                background_color="#FFFFFF",  # Not used since we have a background image
                has_background_image=True,
                background_image_description=f"Edited with instruction: {instruction}",
                nodes=[]  # No nodes - everything is composited into the background
            )

            self.log(f"Saving edited spec to {output_path}")
            saved_path = self.save_spec(edited_spec, output_path)

            # Step 7: Render the final result (just the background)
            self.log("Rendering final result...")
            render_output = saved_path.parent / "render.png"
            try:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        render_image,
                        edited_spec,
                        render_output,
                        edited_spec.canvas_width,
                        edited_spec.canvas_height,
                        saved_path.parent
                    )
                    future.result()
                self.log(f"✓ Rendered to {render_output}")
            except Exception as e:
                self.log(f"Warning: Failed to render: {e}")

            self.log("✓ Edit complete! New spec has no nodes, background saved to background.png")

            return saved_path

        finally:
            # Clean up temporary rendered image
            if temp_render_path.exists():
                temp_render_path.unlink()
