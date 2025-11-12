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

    Limitations: Can only edit images/SVGs via tools. Cannot modify text, layout,
    or other spec properties. For those, use ZeroShotAgent instead.
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

        # Apply edits
        self.log(f"Applying instruction: '{instruction}'")
        self.current_spec_path = output_dir
        edited_spec = self._apply_edit_with_tools(instruction)

        # Save and render
        self.log(f"Saving edited spec to {output_path}")
        saved_path = self.save_spec(edited_spec, output_path)
        self._render_design(saved_path, edited_spec)

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

            # Execute tool calls if any
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                self.log(f"LLM called {len(assistant_message.tool_calls)} tool(s)")

                for tool_call in assistant_message.tool_calls:
                    result = self._execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            else:
                self.log("LLM finished editing")
                break

        return self.current_spec

    def _execute_tool(self, tool_call) -> dict:
        """Route tool call to appropriate handler."""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        self.log(f"Executing {function_name}({function_args})")

        if function_name == "update_image":
            return self._tool_update_image(**function_args)
        elif function_name == "update_svg":
            return self._tool_update_svg(**function_args)
        else:
            return {"error": f"Unknown function: {function_name}"}

    def _build_tool_definitions(self) -> list:
        """Build OpenAI function definitions."""
        return [
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
        return """You are a design editing assistant with access to image and SVG editing tools.

AVAILABLE TOOLS:
- update_image(filename, edit_instruction): Edit image assets using AI
- update_svg(filename, edit_instruction): Regenerate SVG assets using AI

LIMITATIONS:
- You can ONLY modify images and SVGs via tools
- You CANNOT modify text, layout, colors, or other properties
- For those edits, tell the user this agent cannot help

DESIGN SPEC FORMAT:
- nodes: Array with types "text", "image", "svg"
- Each node has: filename, x, y, width, height, rotation, opacity
- Text nodes: text, font-family, font-size, color
- Image nodes: filename, asset_description
- SVG nodes: filename, svg_description

When you call tools, they will be executed and you'll receive results. You can call tools multiple times."""

    def _build_user_prompt(self, instruction: str) -> str:
        """Build user prompt with spec and instruction."""
        spec_json = json.dumps(self.current_spec.model_dump(), indent=2)
        return f"""Current design spec:
```json
{spec_json}
```

User instruction: {instruction}

Analyze the design and use tools to make the requested edits. If the edit requires modifying text/layout/properties, tell the user this agent cannot help with that."""

    # ============ TOOL IMPLEMENTATIONS ============

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
