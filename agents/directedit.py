"""Direct edit agent using function calling to modify images/SVGs with closed-loop captioning."""

import os
import json
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
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

    Tools available:
    - update_spec: Modify text, layout, colors, positions, and all spec properties
    - update_image: Edit image files using AI (Kontext)
    - update_svg: Regenerate SVG files using AI (Gemini)

    Uses iterative function calling to apply edits with verification.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        verbose: bool = False,
        max_iterations: int = 3
    ):
        super().__init__(verbose=verbose)
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations

        # Load API keys
        self.fal_api_key = os.getenv("FAL_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Select appropriate key based on model
        self.api_key = self.gemini_api_key if model.startswith("gemini/") else openai_api_key
        if not self.api_key:
            required = "GEMINI_API_KEY" if model.startswith("gemini/") else "OPENAI_API_KEY"
            raise ValueError(f"{required} environment variable not set")

        # Current state
        self.current_spec_path = None
        self.current_spec = None
        self.source_image_url = None

    def edit(self, spec_path: Path, instruction: str, output_path: Path) -> Path:
        """Edit a design spec using function calling with direct image/SVG editing."""
        # Load spec and assets
        self.log(f"Loading spec from {spec_path}")
        self.current_spec_path = spec_path.parent
        self.current_spec = self.load_spec(spec_path)

        # Load source image for Kontext editing
        self.source_image_url = self._load_source_image(spec_path.parent.name)

        # Copy assets to output directory
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"Copying assets from {self.current_spec_path} to {output_dir}")
        self.copy_assets(self.current_spec_path, output_dir)

        # Initialize edit log
        self.edit_log_path = output_dir / "edit_log.txt"
        self._init_edit_log(instruction)

        # Apply edits
        self.log(f"Applying instruction: '{instruction}'")
        self.current_spec_path = output_dir
        edited_spec = self._apply_edit_with_tools(instruction)

        # Save and render
        self.log(f"Saving edited spec to {output_path}")
        saved_path = self.save_spec(edited_spec, output_path)
        self._render_design(saved_path, edited_spec)

        self._append_to_edit_log("\n" + "="*80 + "\nEDIT COMPLETED\n" + "="*80)

        return saved_path

    def _load_source_image(self, design_name: str) -> str | None:
        """Find and load source image as data URL."""
        candidates = [
            Path(f"datasets/original/{design_name}.jpg"),
            Path(f"datasets/original/{design_name}.png"),
            Path(f"datasets/specs/{design_name}/render.png"),
        ]

        for candidate in candidates:
            if candidate.exists():
                self.log(f"Loading source image from {candidate}")
                return _to_data_url(candidate)

        self.log("Warning: No source image found, image editing may not work")
        return None

    def _render_design(self, saved_path: Path, spec: Spec):
        """Render design to PNG in separate thread to avoid async conflicts."""
        self.log("Rendering edited design...")
        render_output = saved_path.parent / "render.png"

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    render_image, spec, render_output,
                    spec.canvas_width, spec.canvas_height, saved_path.parent
                )
                future.result()
            self.log(f"✓ Rendered to {render_output}")
        except Exception as e:
            self.log(f"Warning: Failed to render: {e}")

    def _apply_edit_with_tools(self, instruction: str) -> Spec:
        """Apply edits using function calling loop."""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(instruction)}
        ]

        tools = self._build_tool_definitions()

        # Iterative function calling loop
        for iteration in range(self.max_iterations):
            self.log(f"Iteration {iteration + 1}/{self.max_iterations}")
            self._append_to_edit_log(f"\n{'='*80}\nITERATION {iteration + 1}/{self.max_iterations}\n{'='*80}\n")

            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                api_key=self.api_key,
                temperature=self.temperature
            )

            assistant_message = response.choices[0].message
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls if hasattr(assistant_message, 'tool_calls') else None
            })

            # Log LLM response
            if assistant_message.content:
                self._append_to_edit_log(f"\nLLM Response:\n{assistant_message.content}\n")

            # Execute tool calls if any
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                self.log(f"LLM called {len(assistant_message.tool_calls)} tool(s)")
                self._append_to_edit_log(f"\nTool Calls: {len(assistant_message.tool_calls)}\n")

                for tool_call in assistant_message.tool_calls:
                    result = self._execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            else:
                self.log("LLM finished editing")
                self._append_to_edit_log("\nLLM finished editing (no more tool calls)\n")
                break

        return self.current_spec

    def _execute_tool(self, tool_call) -> dict:
        """Route tool call to appropriate handler."""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        self.log(f"Executing {function_name}(...)")

        # Log with truncated args for readability
        if function_name == "update_spec":
            self._append_to_edit_log(f"\n  → {function_name}(modified_spec={{...}})\n")
        else:
            self._append_to_edit_log(f"\n  → {function_name}({json.dumps(function_args, indent=4)})\n")

        if function_name == "update_spec":
            result = self._tool_update_spec(**function_args)
        elif function_name == "update_image":
            result = self._tool_update_image(**function_args)
        elif function_name == "update_svg":
            result = self._tool_update_svg(**function_args)
        else:
            result = {"error": f"Unknown function: {function_name}"}

        # Log result
        self._append_to_edit_log(f"  ← Result: {json.dumps(result, indent=4)}\n")

        return result

    def _build_tool_definitions(self) -> list:
        """Build OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "update_spec",
                    "description": "Update the design spec JSON to modify text, layout, colors, positions, or any other properties. Returns the complete modified spec.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "modified_spec": {
                                "type": "object",
                                "description": "The complete modified design spec as JSON. Must include all required fields: canvas_width, canvas_height, background_color, has_background_image, nodes array."
                            }
                        },
                        "required": ["modified_spec"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_image",
                    "description": "Edit an image asset using AI image editing (Kontext). The image will be extracted from the source design with modifications applied.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Image filename (e.g., 'asset-1.png')"
                            },
                            "edit_instruction": {
                                "type": "string",
                                "description": "How to edit the image (e.g., 'Change the car from red to blue')"
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
                    "description": "Regenerate an SVG asset using AI (Gemini). The SVG will be generated from scratch.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "SVG filename (e.g., 'svg-1.svg')"
                            },
                            "edit_instruction": {
                                "type": "string",
                                "description": "How to modify the SVG (e.g., 'Make the icon blue')"
                            }
                        },
                        "required": ["filename", "edit_instruction"]
                    }
                }
            }
        ]

    def _build_system_prompt(self) -> str:
        """Build system prompt."""
        return """You are a design editing assistant with access to three powerful tools for modifying designs.

AVAILABLE TOOLS:

1. update_spec(modified_spec)
   - Use this to modify TEXT, LAYOUT, COLORS, POSITIONS, or any SPEC PROPERTIES
   - Pass the COMPLETE modified spec JSON with all your changes
   - Use for: changing text content, colors, fonts, positions, spacing, adding/removing nodes, etc.
   - This is your PRIMARY tool for most edits

2. update_image(filename, edit_instruction)
   - Use this to modify IMAGE FILES using AI (changes pixels, not spec properties)
   - Only use when you need to change the actual image content (e.g., "make the car blue", "change photo to mountains")
   - Do NOT use for positioning, sizing, or opacity - use update_spec instead

3. update_svg(filename, edit_instruction)
   - Use this to regenerate SVG FILES from scratch
   - Only use when you need to change the SVG artwork itself
   - Do NOT use for positioning, sizing, or opacity - use update_spec instead

DESIGN SPEC FORMAT:
The spec is a JSON object with:
- canvas_width, canvas_height: Canvas dimensions (integers)
- background_color: Background color (hex string like "#ffffff")
- has_background_image: Boolean indicating if there's a background image
- background_image_description: Optional string describing the background
- nodes: Array of design elements, each with:
  - type: "text", "image", or "svg"
  - x, y: Position (float)
  - width, height: Size (float)
  - rotation: Rotation in degrees (float, default 0)
  - opacity: Opacity 0-1 (float, default 1)

  TEXT nodes also have:
  - text: The text content (string)
  - font_family: Font name (string)
  - font_size: Font size in px (float)
  - color: Text color (hex string)
  - text_align: "left", "center", or "right"
  - font_weight: "normal", "bold", or numeric (string)
  - font_style: "normal" or "italic"
  - text_decoration: "none" or "underline"
  - text_transform: "none", "uppercase", "lowercase", "capitalize"

  IMAGE nodes also have:
  - filename: Image file (string, e.g., "asset-1.png")
  - asset_description: Description of the image (string)

  SVG nodes also have:
  - filename: SVG file (string, e.g., "svg-1.svg")
  - svg_description: Description of the SVG (string)

WORKFLOW:
1. Analyze the current spec and the user's instruction
2. Determine which tool(s) to use:
   - For text/layout/color/position changes → update_spec
   - For changing image content → update_image
   - For regenerating SVG artwork → update_svg
3. Call the appropriate tool(s)
4. You can call tools multiple times if needed

IMPORTANT:
- When using update_spec, return the COMPLETE spec with ALL fields
- Preserve all nodes you're not modifying
- Make sure your JSON is valid
- You can combine tools (e.g., update_spec + update_image) in multiple iterations"""

    def _build_user_prompt(self, instruction: str) -> str:
        """Build user prompt with spec and instruction."""
        spec_json = json.dumps(self.current_spec.model_dump(), indent=2)
        return f"""Current design spec:
```json
{spec_json}
```

User instruction: {instruction}

Analyze the current spec and the user's instruction. Determine which tool(s) to use and call them to make the requested edits."""

    # ============ TOOL IMPLEMENTATIONS ============

    def _tool_update_spec(self, modified_spec: dict) -> dict:
        """Update the design spec with modifications."""
        try:
            self.log("Applying spec modifications...")

            # Validate the modified spec using Pydantic
            try:
                new_spec = Spec.model_validate(modified_spec)
            except Exception as e:
                return {"error": f"Invalid spec format: {str(e)}"}

            # Update the current spec
            self.current_spec = new_spec
            self.log("✓ Spec updated successfully")

            return {
                "success": True,
                "message": "Spec updated successfully",
                "nodes_count": len(new_spec.nodes)
            }

        except Exception as e:
            self.log(f"Error updating spec: {e}")
            return {"error": str(e)}

    def _tool_update_image(self, filename: str, edit_instruction: str) -> dict:
        """Edit an image asset using Kontext."""
        try:
            if not self.source_image_url:
                return {"error": "No source image available for editing"}

            image_path = self.current_spec_path / filename
            self.log(f"Editing {filename}: {edit_instruction}")

            # Use Kontext to edit
            prompt = f"Isolate and extract this element with modifications: {edit_instruction}"
            success = self._run_async(self._edit_image_async(prompt, image_path))

            if not success:
                return {"error": f"Failed to generate edited image for {filename}"}

            self.log(f"✓ Saved edited image to {filename}")

            # Caption for verification
            self.log(f"Captioning edited {filename}...")
            new_caption = caption_image(image_path)

            if not new_caption:
                return {"error": f"Failed to caption {filename}"}

            # Update spec
            self._update_node_property(filename, 'asset_description', new_caption)

            return {
                "success": True,
                "filename": filename,
                "new_caption": new_caption,
                "message": f"Successfully edited {filename}"
            }

        except Exception as e:
            self.log(f"Error editing {filename}: {e}")
            return {"error": str(e)}

    def _tool_update_svg(self, filename: str, edit_instruction: str) -> dict:
        """Regenerate an SVG asset using Gemini."""
        try:
            svg_path = self.current_spec_path / filename
            self.log(f"Regenerating {filename}: {edit_instruction}")

            # Get current description and append modification
            current_desc = self._get_node_property(filename, 'svg_description', '')
            new_desc = f"{current_desc}. Modified: {edit_instruction}" if current_desc else edit_instruction

            # Generate SVG
            svg_content = self._run_async(self._generate_svg_async(new_desc))
            svg_path.write_text(svg_content, encoding='utf-8')

            self.log(f"✓ Saved regenerated SVG to {filename}")

            # Update spec
            self._update_node_property(filename, 'svg_description', new_desc)

            return {
                "success": True,
                "filename": filename,
                "updated_description": new_desc,
                "message": f"Successfully regenerated {filename}"
            }

        except Exception as e:
            self.log(f"Error regenerating {filename}: {e}")
            return {"error": str(e)}

    # ============ HELPER METHODS ============

    def _init_edit_log(self, instruction: str):
        """Initialize edit log file."""
        from datetime import datetime

        header = f"""{'='*80}
DIRECTEDIT AGENT EDIT LOG
{'='*80}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {self.model}
Temperature: {self.temperature}
Max Iterations: {self.max_iterations}

USER INSTRUCTION:
{instruction}

{'='*80}
"""
        self.edit_log_path.write_text(header, encoding='utf-8')

    def _append_to_edit_log(self, content: str):
        """Append content to edit log."""
        if hasattr(self, 'edit_log_path') and self.edit_log_path:
            with open(self.edit_log_path, 'a', encoding='utf-8') as f:
                f.write(content)

    def _update_node_property(self, filename: str, property_name: str, value: any):
        """Update a property on a node by filename."""
        for node in self.current_spec.nodes:
            if hasattr(node, 'filename') and node.filename == filename:
                setattr(node, property_name, value)
                self.log(f"Updated {filename}.{property_name}: {value}")
                break

    def _get_node_property(self, filename: str, property_name: str, default: any = None) -> any:
        """Get a property from a node by filename."""
        for node in self.current_spec.nodes:
            if hasattr(node, 'filename') and node.filename == filename:
                return getattr(node, property_name, default)
        return default

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        return asyncio.run(coro)

    async def _edit_image_async(self, prompt: str, image_path: Path) -> bool:
        """Edit image using Kontext API."""
        timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_read=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            result = await self._kontext_edit_async(prompt, self.source_image_url, session)

            if 'images' in result and result['images']:
                image_url = result['images'][0]['url']
                await self._download_image(image_url, image_path, session)
                return True
            return False

    async def _kontext_edit_async(self, prompt: str, image_url: str, session: aiohttp.ClientSession) -> dict:
        """Call Kontext API for context-aware image editing."""
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
                # Poll for completion
                job = await response.json()
                status_url = job.get("status_url") or job.get("response_url")

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
        """Generate SVG markup using Gemini."""
        prompt = f"""Generate clean, minimal SVG markup for: {description}

Requirements:
- Return ONLY the SVG markup, no explanations
- Use viewBox for scalability
- Keep it simple and clean

SVG:"""

        response = await litellm.acompletion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            api_key=self.gemini_api_key,
            temperature=0.3,
            drop_params=True,
        )

        svg_content = response.choices[0].message.content.strip()

        # Clean up code fences
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
