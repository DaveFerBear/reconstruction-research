"""Direct edit agent using function calling to modify images/SVGs with closed-loop captioning."""

import os
import json
import asyncio
from pathlib import Path
import litellm
import aiohttp
from dotenv import load_dotenv

from .base import Agent
from lib.types import Spec
from lib.caption import caption_image
from lib.utils import _to_data_url
from lib.render import render_image

load_dotenv()


class DirectEditAgent(Agent):
    """
    Direct edit agent with function calling.

    Uses tools to directly edit images/SVGs, then captions them to verify changes.
    This creates a closed-loop system where edits are applied and then described
    back to the LLM for verification or further refinement.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        verbose: bool = False,
        max_iterations: int = 10
    ):
        """
        Initialize the direct edit agent.

        Args:
            model: LiteLLM model identifier (default: gpt-4o for function calling)
            temperature: Sampling temperature for generation (0.0-1.0)
            verbose: If True, print debug information
            max_iterations: Maximum number of tool call iterations (default: 10)
        """
        super().__init__(verbose=verbose)
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.fal_api_key = os.getenv("FAL_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Store current spec path for tool access
        self.current_spec_path = None
        self.current_spec = None
        self.source_image_url = None  # Data URL of the original design image

    def edit(self, spec_path: Path, instruction: str, output_path: Path) -> Path:
        """
        Edit a design spec using function calling with direct image/SVG editing.

        Args:
            spec_path: Path to the input design spec (spec.json)
            instruction: Natural language instruction for the edit
            output_path: Path where the edited spec should be saved

        Returns:
            Path: The output path where the edited spec was saved
        """
        self.log(f"Loading spec from {spec_path}")
        self.current_spec_path = spec_path.parent  # Store directory for asset access
        self.current_spec = self.load_spec(spec_path)

        # Load source image for Kontext editing
        # Look for original source image (e.g., in datasets/original/)
        design_name = spec_path.parent.name
        source_candidates = [
            Path(f"datasets/original/{design_name}.jpg"),
            Path(f"datasets/original/{design_name}.png"),
            spec_path.parent / "render.png",  # Fallback to rendered version
        ]

        source_image_path = None
        for candidate in source_candidates:
            if candidate.exists():
                source_image_path = candidate
                break

        if source_image_path:
            self.log(f"Loading source image from {source_image_path}")
            self.source_image_url = _to_data_url(source_image_path)
        else:
            self.log("Warning: No source image found, image editing may not work")

        self.log(f"Applying instruction: '{instruction}'")
        edited_spec = self._apply_edit_with_tools(instruction)

        self.log(f"Saving edited spec to {output_path}")
        saved_path = self.save_spec(edited_spec, output_path)

        # Render the edited design
        self.log("Rendering edited design...")
        render_output = saved_path.parent / "render.png"
        try:
            render_image(
                edited_spec,
                render_output,
                canvas_width=edited_spec.canvas_width,
                canvas_height=edited_spec.canvas_height,
                asset_dir=saved_path.parent
            )
            self.log(f"✓ Rendered to {render_output}")
        except Exception as e:
            self.log(f"Warning: Failed to render: {e}")

        return saved_path

    def _apply_edit_with_tools(self, instruction: str) -> Spec:
        """
        Apply an edit instruction using function calling tools.

        Args:
            instruction: Natural language instruction

        Returns:
            Spec: Modified design specification
        """
        # Build initial messages
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt()
            },
            {
                "role": "user",
                "content": self._build_user_prompt(instruction)
            }
        ]

        # Tool definitions
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "update_image",
                    "description": "Edit an image asset using AI image editing. The image will be extracted from the source design with your modifications applied, saved to disk, and automatically captioned to verify the changes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "The filename of the image to edit (e.g., 'asset-1.png', 'asset-2.png')"
                            },
                            "edit_instruction": {
                                "type": "string",
                                "description": "Clear instruction for how to edit the image (e.g., 'Change the car from red to blue', 'Make the dog a golden retriever instead')"
                            }
                        },
                        "required": ["filename", "edit_instruction"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_svg",
                    "description": "Regenerate an SVG asset using AI. The SVG will be generated from scratch based on your description, saved to disk, and the spec will be updated.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "The filename of the SVG to update (e.g., 'svg-1.svg', 'svg-2.svg')"
                            },
                            "edit_instruction": {
                                "type": "string",
                                "description": "Clear instruction for how to modify the SVG (e.g., 'Make the icon blue instead of red', 'Add a border around the shape')"
                            }
                        },
                        "required": ["filename", "edit_instruction"]
                    }
                }
            }
        ]

        # Iteratively call LLM with function calling
        for iteration in range(self.max_iterations):
            self.log(f"Iteration {iteration + 1}/{self.max_iterations}")

            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                api_key=self.api_key,
                temperature=self.temperature
            )

            assistant_message = response.choices[0].message

            # Add assistant message to history
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls if hasattr(assistant_message, 'tool_calls') else None
            })

            # Check if LLM wants to call tools
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                self.log(f"LLM called {len(assistant_message.tool_calls)} tool(s)")

                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    self.log(f"Executing {function_name}({function_args})")

                    # Execute the function
                    if function_name == "update_image":
                        result = self._tool_update_image(**function_args)
                    elif function_name == "update_svg":
                        result = self._tool_update_svg(**function_args)
                    else:
                        result = {"error": f"Unknown function: {function_name}"}

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })

            else:
                # LLM is done, no more tool calls
                self.log("LLM finished editing")
                break

        return self.current_spec

    def _tool_update_image(self, filename: str, edit_instruction: str) -> dict:
        """
        Tool handler: Edit an image asset.

        Args:
            filename: Image filename (e.g., 'asset-1.png')
            edit_instruction: How to edit the image

        Returns:
            dict: Result with new caption or error
        """
        try:
            image_path = self.current_spec_path / filename

            if not self.source_image_url:
                return {"error": "No source image available for editing"}

            self.log(f"Editing {filename}: {edit_instruction}")

            # Use Kontext to edit the image
            prompt = f"Isolate and extract this element with modifications: {edit_instruction}"

            # Run async function in sync context
            async def edit_async():
                timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_read=60)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # Edit with Kontext
                    result = await self._kontext_edit_async(prompt, self.source_image_url, session)

                    if 'images' in result and result['images']:
                        image_url = result['images'][0]['url']
                        await self._download_image(image_url, image_path, session)
                        return True
                    return False

            success = asyncio.run(edit_async())

            if not success:
                return {"error": f"Failed to generate edited image for {filename}"}

            self.log(f"✓ Saved edited image to {filename}")

            # Caption the edited image
            self.log(f"Captioning edited {filename}...")
            new_caption = caption_image(image_path)

            if not new_caption:
                return {"error": f"Failed to caption {filename}"}

            # Update the spec
            for node in self.current_spec.nodes:
                if hasattr(node, 'filename') and node.filename == filename:
                    node.asset_description = new_caption
                    self.log(f"Updated {filename} description: {new_caption}")
                    break

            return {
                "success": True,
                "filename": filename,
                "new_caption": new_caption,
                "message": f"Successfully edited and saved {filename}"
            }

        except Exception as e:
            self.log(f"Error editing {filename}: {e}")
            return {"error": str(e)}

    def _tool_update_svg(self, filename: str, edit_instruction: str) -> dict:
        """
        Tool handler: Regenerate an SVG asset.

        Args:
            filename: SVG filename (e.g., 'svg-1.svg')
            edit_instruction: How to modify the SVG

        Returns:
            dict: Result with updated description or error
        """
        try:
            svg_path = self.current_spec_path / filename

            self.log(f"Regenerating {filename}: {edit_instruction}")

            # Find the node to get current description
            current_desc = ""
            for node in self.current_spec.nodes:
                if hasattr(node, 'filename') and node.filename == filename:
                    current_desc = getattr(node, 'svg_description', '')
                    break

            # Build new description incorporating the edit
            if current_desc:
                new_desc = f"{current_desc}. Modified: {edit_instruction}"
            else:
                new_desc = edit_instruction

            # Generate SVG using Gemini
            async def generate_async():
                svg_content = await self._generate_svg_async(new_desc)
                svg_path.write_text(svg_content, encoding='utf-8')
                return svg_content

            svg_content = asyncio.run(generate_async())

            self.log(f"✓ Saved regenerated SVG to {filename}")

            # Update the spec's svg_description
            for node in self.current_spec.nodes:
                if hasattr(node, 'filename') and node.filename == filename:
                    node.svg_description = new_desc
                    self.log(f"Updated {filename} description: {new_desc}")
                    break

            return {
                "success": True,
                "filename": filename,
                "updated_description": new_desc,
                "message": f"Successfully regenerated {filename}"
            }

        except Exception as e:
            self.log(f"Error regenerating {filename}: {e}")
            return {"error": str(e)}

    def _build_system_prompt(self) -> str:
        """Build the system prompt explaining the design spec format and tools."""
        return """You are a design editing assistant with direct access to image and SVG editing tools.

You can view a design specification in JSON format and use tools to modify it.

AVAILABLE TOOLS:
- update_image(filename, edit_instruction): Edit an image asset using AI image editing (Kontext)
- update_svg(filename, edit_instruction): Regenerate an SVG asset using AI (Gemini)

When you call these tools:
1. The asset will be ACTUALLY MODIFIED according to your instruction (not simulated!)
2. For images: Uses Kontext AI to extract/modify the element from the source design
3. For SVGs: Uses Gemini to generate new SVG markup based on your description
4. The modified asset is saved to disk
5. Images are automatically captioned to verify the changes
6. The spec is updated with the new description
7. You'll receive the result to confirm or make further edits

DESIGN SPEC FORMAT:
- canvas_width, canvas_height: Canvas dimensions
- background_color: Background color (hex format)
- has_background_image: Whether there's a background image
- nodes: Array of design elements with types: "text", "image", "svg"

For image nodes:
- filename: The image file (e.g., "asset-1.png")
- asset_description: Current description of the image
- x, y, width, height: Position and size
- rotation, opacity: Visual properties

For SVG nodes:
- filename: The SVG file (e.g., "svg-1.svg")
- svg_description: Current description of the SVG
- x, y, width, height: Position and size
- rotation, opacity: Visual properties

For text nodes:
- text: The text content
- font-family, font-size, color: Typography
- x, y, width, height: Position and size

IMPORTANT:
- Use tools to modify images/SVGs, don't try to return modified JSON
- You can modify text nodes and canvas properties directly in your response
- Be specific in your edit instructions to the tools
- After tools complete, you can make additional edits or confirm completion"""

    def _build_user_prompt(self, instruction: str) -> str:
        """Build the user prompt with the current spec and instruction."""
        spec_json = json.dumps(self.current_spec.model_dump(), indent=2)

        return f"""Current design spec:
```json
{spec_json}
```

User instruction: {instruction}

Please analyze the design and use the available tools to make the requested edits. When you're done, respond with a summary of what you changed."""

    def _extract_json(self, response_text: str) -> dict:
        """
        Extract JSON from LLM response (fallback if not using tools).

        Args:
            response_text: Raw response from LLM

        Returns:
            dict: Parsed JSON object
        """
        # Try to find JSON in code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        else:
            json_str = response_text.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}\n\nResponse:\n{response_text}")

    async def _kontext_edit_async(self, prompt: str, image_url: str, session: aiohttp.ClientSession) -> dict:
        """Async Kontext edit for context-aware image extraction/modification."""
        headers = {"Authorization": f"Key {self.fal_api_key}"}
        payload = {"prompt": prompt, "image_url": image_url}

        async with session.post("https://fal.run/fal-ai/flux-pro/kontext", json=payload, headers=headers) as response:
            if response.status == 422:
                error_text = await response.text()
                raise Exception(f"Kontext API Error: {error_text}")
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
                            raise RuntimeError(f"Kontext job failed: {data}")

    async def _generate_svg_async(self, description: str) -> str:
        """Generate SVG markup from a text description using Gemini."""
        prompt = f"""Generate clean, minimal SVG markup for: {description}

Requirements:
- Return ONLY the SVG markup, no explanations or code fences
- Use viewBox for scalability
- Keep it simple and clean
- Use appropriate colors mentioned in the description
- Make it production-ready

SVG:"""

        response = await litellm.acompletion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            api_key=self.gemini_api_key,
            temperature=0.3,
            drop_params=True,
        )

        svg_content = response.choices[0].message.content.strip()

        # Clean up if model added code fences
        if svg_content.startswith("```"):
            lines = svg_content.split('\n')
            svg_content = '\n'.join(lines[1:-1]) if len(lines) > 2 else svg_content
            svg_content = svg_content.replace("```svg", "").replace("```", "").strip()

        return svg_content

    async def _download_image(self, url: str, path: Path, session: aiohttp.ClientSession):
        """Download image from URL to path."""
        async with session.get(url) as response:
            response.raise_for_status()
            content = await response.read()
            path.write_bytes(content)
