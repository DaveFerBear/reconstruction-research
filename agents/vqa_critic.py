"""Two-agent system: Critic provides feedback, Editor implements changes."""

import os
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import litellm
from dotenv import load_dotenv
from typing import Optional

from .base import Agent
from lib.types import Spec
from lib.utils import _to_data_url
from lib.render import render_image
from lib.ai import edit_image_local

load_dotenv()


class VQACriticAgent(Agent):
    """
    Two-agent system with critic and editor.

    Critic: Reviews the design and provides feedback or marks as complete.
    Editor: Makes edits based on critic's feedback using tool calling.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        verbose: bool = False,
        max_iterations: int = 5,
        image_editor: Optional[str] = None,
        image_edit_size: Optional[str] = None,
        save_renders: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.save_renders = save_renders

        # Load API keys
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Select appropriate key based on model
        self.api_key = self.gemini_api_key if model.startswith("gemini/") else openai_api_key
        if not self.api_key:
            required = "GEMINI_API_KEY" if model.startswith("gemini/") else "OPENAI_API_KEY"
            raise ValueError(f"{required} environment variable not set")

        # Image editor settings
        self.image_editor = image_editor or "gpt-image-1"
        self.image_edit_size = image_edit_size

        # Current state
        self.current_spec_path = None
        self.current_spec = None

    def edit(self, spec_path: Path, instruction: str, output_path: Path) -> Path:
        """Edit a design using critic-editor loop."""
        # Load spec and assets
        self.log(f"Loading spec from {spec_path}")
        self.current_spec_path = spec_path.parent
        self.current_spec = self.load_spec(spec_path)

        # Copy assets to output directory
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"Copying assets from {self.current_spec_path} to {output_dir}")
        self.copy_assets(self.current_spec_path, output_dir)

        # Initialize edit log
        self.edit_log_path = output_dir / "edit_log.txt"
        self._init_edit_log(instruction)

        # Update working directory
        self.current_spec_path = output_dir

        # First edit: Editor implements the original user instruction
        self.log(f"ITERATION 1/{self.max_iterations}")
        self._append_to_edit_log(f"\n{'='*80}\nITERATION 1\n{'='*80}\n")
        self.log("Editor implementing original instruction...")
        self._append_to_edit_log(f"\nEDITOR IMPLEMENTING: {instruction}\n")
        self._editor_make_changes(instruction)

        # Critic-editor loop
        final_iteration = 0
        for iteration in range(self.max_iterations):
            self.log(f"Iteration {iteration + 1}/{self.max_iterations} - Critic review")

            # Save and render current state
            saved_path = self.save_spec(self.current_spec, output_path)
            render_url = self._render_design(saved_path, self.current_spec, iteration)

            # Critic: Review the design
            feedback = self._critic_review(instruction, render_url, iteration)
            critic_instruction = feedback.get('instruction', '')
            self._append_to_edit_log(f"\nCRITIC EVALUATION:\n")
            self._append_to_edit_log(f"  Complete: {feedback['complete']}\n")
            self._append_to_edit_log(f"  Instruction: {critic_instruction}\n")
            self.log(f"Critic complete={feedback['complete']}: {critic_instruction}")

            # Check if complete
            if feedback['complete']:
                self.log("✓ Critic says design is complete!")
                final_iteration = iteration
                break

            # Editor: Implement critic's instruction (if not at max iterations)
            if iteration + 1 < self.max_iterations:
                self.log(f"ITERATION {iteration + 2}/{self.max_iterations}")
                self._append_to_edit_log(f"\n{'='*80}\nITERATION {iteration + 2}\n{'='*80}\n")
                self.log("Editor implementing critic's instruction...")
                self._append_to_edit_log(f"\nEDITOR IMPLEMENTING: {critic_instruction}\n")
                self._editor_make_changes(critic_instruction)
            final_iteration = iteration + 1

        # Final save and render
        self._append_to_edit_log("\n" + "="*80 + "\nEDIT COMPLETED\n" + "="*80)
        saved_path = self.save_spec(self.current_spec, output_path)

        # Create final render.png
        self._create_final_render(saved_path, self.current_spec, final_iteration)

        return saved_path

    def _render_design(self, saved_path: Path, spec: Spec, iteration: int) -> str | None:
        """Render design to PNG and return data URL. Saves intermediate renders if enabled."""
        self.log("Rendering design...")

        # Use iteration-specific filename if save_renders is True
        if self.save_renders:
            render_output = saved_path.parent / f"render_iter_{iteration}.png"
        else:
            # Use temporary file that will be overwritten
            render_output = saved_path.parent / "render_temp.png"

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    render_image, spec, render_output,
                    spec.canvas_width, spec.canvas_height, saved_path.parent
                )
                future.result()
            self.log(f"✓ Rendered to {render_output}")
            return _to_data_url(render_output)
        except Exception as e:
            self.log(f"Warning: Failed to render: {e}")
            return None

    def _create_final_render(self, saved_path: Path, spec: Spec, final_iteration: int):
        """Create the final render.png file."""
        self.log("Creating final render...")
        render_output = saved_path.parent / "render.png"

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    render_image, spec, render_output,
                    spec.canvas_width, spec.canvas_height, saved_path.parent
                )
                future.result()
            self.log(f"✓ Final render saved to {render_output}")

            # If save_renders is False, clean up temporary file
            if not self.save_renders:
                temp_file = saved_path.parent / "render_temp.png"
                if temp_file.exists():
                    temp_file.unlink()
        except Exception as e:
            self.log(f"Warning: Failed to create final render: {e}")
            # Try to copy from last iteration if it exists
            if self.save_renders:
                last_render = saved_path.parent / f"render_iter_{final_iteration}.png"
                if last_render.exists():
                    import shutil
                    shutil.copy(last_render, render_output)
                    self.log(f"✓ Copied from {last_render}")

    def _critic_review(self, instruction: str, render_url: str | None, iteration: int) -> dict:
        """Critic reviews the design and provides feedback."""
        spec_json = json.dumps(self.current_spec.model_dump(), indent=2)

        content = [
            {
                "type": "text",
                "text": f"""You are a design critic. Evaluate this design on two dimensions:

1. **Edit Success**: Does it fulfill the user's original instruction?
2. **Design Quality**: Evaluate overall design elements including:
   - Visual balance and composition
   - Alignment and spacing
   - Color harmony
   - Typography hierarchy
   - Overall aesthetic quality

Original user instruction: {instruction}

Current design spec:
```json
{spec_json}
```

Current rendered design:"""
            }
        ]

        if render_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": render_url}
            })

        content.append({
            "type": "text",
            "text": """
Provide your evaluation in JSON format:
{
  "complete": true/false,
  "instruction": "new instruction for the editor"
}

**If complete=false**: Provide a clear, actionable instruction addressing the most important issue. This could be:
- Completing/fixing the original edit instruction
- Fixing design quality issues (alignment, spacing, balance, colors, etc.)
- Any combination of the above

**If complete=true**: Set instruction to an empty string. The design meets the original instruction AND has good design quality.

Focus on ONE clear instruction per iteration for best results."""
        })

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            api_key=self.api_key,
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )

        text = response.choices[0].message.content.strip()
        result = json.loads(text)

        # Normalize the response format (support both "message" and "instruction")
        if "message" in result and "instruction" not in result:
            result["instruction"] = result["message"]

        return result

    def _editor_make_changes(self, instruction: str):
        """Editor makes changes based on critic instruction using tool calling."""
        spec_json = json.dumps(self.current_spec.model_dump(), indent=2)

        messages = [
            {
                "role": "system",
                "content": """You are a design editor. Implement the critic's instruction using the available tools.

AVAILABLE TOOLS:
1. update_spec(modified_spec) - Modify text, layout, colors, positions, or any spec properties
2. update_image(filename, edit_instruction) - Edit image files using AI
3. update_svg(filename, edit_instruction) - Regenerate SVG files

Use update_spec for most changes (text, layout, colors, positions, etc.)."""
            },
            {
                "role": "user",
                "content": f"""Current design spec:
```json
{spec_json}
```

Critic's instruction: {instruction}

Implement this instruction using the appropriate tools."""
            }
        ]

        tools = self._build_tool_definitions()

        response = litellm.completion(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            api_key=self.api_key,
            temperature=self.temperature
        )

        assistant_message = response.choices[0].message

        # Execute tool calls
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            self.log(f"Editor called {len(assistant_message.tool_calls)} tool(s)")
            self._append_to_edit_log(f"\nEDITOR TOOL CALLS: {len(assistant_message.tool_calls)}\n")

            for tool_call in assistant_message.tool_calls:
                result = self._execute_tool(tool_call)
                self._append_to_edit_log(f"  � {tool_call.function.name}: {json.dumps(result, indent=4)}\n")
        else:
            self.log("Editor made no tool calls")
            self._append_to_edit_log("\nEditor made no changes\n")

    def _execute_tool(self, tool_call) -> dict:
        """Route tool call to appropriate handler."""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        self.log(f"Executing {function_name}(...)")

        if function_name == "update_spec":
            return self._tool_update_spec(**function_args)
        elif function_name == "update_image":
            return self._tool_update_image(**function_args)
        elif function_name == "update_svg":
            return self._tool_update_svg(**function_args)
        else:
            return {"error": f"Unknown function: {function_name}"}

    def _build_tool_definitions(self) -> list:
        """Build tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "update_spec",
                    "description": "Update the design spec JSON to modify text, layout, colors, positions, or any other properties.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "modified_spec": {
                                "type": "object",
                                "description": "The complete modified design spec as JSON."
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
                    "description": "Edit an image asset using AI image editing.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Image filename"},
                            "edit_instruction": {"type": "string", "description": "How to edit the image"}
                        },
                        "required": ["filename", "edit_instruction"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_svg",
                    "description": "Regenerate an SVG asset using AI.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "SVG filename"},
                            "edit_instruction": {"type": "string", "description": "How to modify the SVG"}
                        },
                        "required": ["filename", "edit_instruction"]
                    }
                }
            }
        ]

    # ============ TOOL IMPLEMENTATIONS ============

    def _tool_update_spec(self, modified_spec: dict) -> dict:
        """Update the design spec."""
        try:
            new_spec = Spec.model_validate(modified_spec)
            self.current_spec = new_spec
            self.log(" Spec updated successfully")
            return {"success": True, "message": "Spec updated successfully"}
        except Exception as e:
            self.log(f"Error updating spec: {e}")
            return {"error": str(e)}

    def _tool_update_image(self, filename: str, edit_instruction: str) -> dict:
        """Edit an image asset."""
        try:
            image_path = self.current_spec_path / filename
            self.log(f"Editing {filename}: {edit_instruction}")

            edited_bytes = edit_image_local(edit_instruction, image_path, backend=self.image_editor, size=self.image_edit_size)
            image_path.write_bytes(edited_bytes)

            self.log(f" Saved edited image to {filename}")
            return {"success": True, "filename": filename, "message": f"Successfully edited {filename}"}
        except Exception as e:
            self.log(f"Error editing {filename}: {e}")
            return {"error": str(e)}

    def _tool_update_svg(self, filename: str, edit_instruction: str) -> dict:
        """Regenerate an SVG asset."""
        try:
            import asyncio
            svg_path = self.current_spec_path / filename
            self.log(f"Regenerating {filename}: {edit_instruction}")

            # Simple SVG generation using Gemini
            async def generate():
                response = await litellm.acompletion(
                    model="gemini/gemini-2.5-flash",
                    messages=[{"role": "user", "content": f"Generate clean SVG markup for: {edit_instruction}\n\nReturn ONLY the SVG markup."}],
                    api_key=self.gemini_api_key,
                    temperature=0.3,
                    drop_params=True,
                )
                svg_content = response.choices[0].message.content.strip()
                if svg_content.startswith("```"):
                    lines = svg_content.split('\n')
                    svg_content = '\n'.join(lines[1:-1]) if len(lines) > 2 else svg_content
                    svg_content = svg_content.replace("```svg", "").replace("```", "").strip()
                return svg_content

            svg_content = asyncio.run(generate())
            svg_path.write_text(svg_content, encoding='utf-8')

            self.log(f" Saved regenerated SVG to {filename}")
            return {"success": True, "filename": filename, "message": f"Successfully regenerated {filename}"}
        except Exception as e:
            self.log(f"Error regenerating {filename}: {e}")
            return {"error": str(e)}

    # ============ HELPER METHODS ============

    def _init_edit_log(self, instruction: str):
        """Initialize edit log file."""
        from datetime import datetime

        header = f"""{'='*80}
VQA CRITIC AGENT EDIT LOG (Critic-Editor Loop)
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
