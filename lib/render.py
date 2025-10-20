from pathlib import Path
from typing import Union
import json
from playwright.sync_api import sync_playwright
from .types import Spec, TextNode, ImageNode


def _generate_html(spec: Spec, canvas_width: int = 800, canvas_height: int = 600) -> str:
    """Generate HTML from a spec."""

    # Build node HTML
    nodes_html = []
    for node in spec.nodes:
        if isinstance(node, TextNode):
            style = (
                f"position: absolute; "
                f"left: {node.x}px; "
                f"top: {node.y}px; "
                f"width: {node.width}px; "
                f"height: {node.height}px; "
                f"transform: rotate({node.rotation}deg); "
                f"font-family: {node.font_family}; "
                f"font-size: {node.font_size}px; "
                f"color: {node.color}; "
                f"text-align: {node.text_align}; "
                f"font-weight: {node.font_weight}; "
                f"font-style: {node.font_style}; "
                f"text-decoration: {node.text_decoration}; "
                f"text-transform: {node.text_transform}; "
                f"margin: 0; "
                f"padding: 0; "
                f"box-sizing: border-box; "
                f"display: flex; "
                f"align-items: center; "
            )
            # Escape HTML in text content
            text_content = node.text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            nodes_html.append(f'<div style="{style}">{text_content}</div>')

        elif isinstance(node, ImageNode):
            style = (
                f"position: absolute; "
                f"left: {node.x}px; "
                f"top: {node.y}px; "
                f"width: {node.width}px; "
                f"height: {node.height}px; "
                f"transform: rotate({node.rotation}deg); "
                f"background: #ddd; "
                f"display: flex; "
                f"align-items: center; "
                f"justify-content: center; "
                f"font-size: 12px; "
                f"color: #666; "
                f"text-align: center; "
                f"padding: 10px; "
                f"box-sizing: border-box; "
            )
            # For now, show placeholder with description
            desc = node.asset_description[:100]  # Truncate long descriptions
            nodes_html.append(f'<div style="{style}" title="{node.asset_description}">[Image: {desc}]</div>')

    # Build background style
    bg_style = f"background-color: {spec.background_color};"
    if spec.has_background_image and spec.background_image_description:
        # For now, just show the background color since we don't have actual images
        # In a real implementation, you'd generate or fetch the background image
        pass

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
        }}
        #canvas {{
            position: relative;
            width: {canvas_width}px;
            height: {canvas_height}px;
            {bg_style}
            overflow: hidden;
        }}
    </style>
</head>
<body>
    <div id="canvas">
        {''.join(nodes_html)}
    </div>
</body>
</html>"""
    return html


def render_image(
    spec: Union[Spec, dict],
    output_path: Path,
    canvas_width: int = 800,
    canvas_height: int = 600,
) -> Path:
    """
    Render an image from a spec using a headless browser.

    Args:
        spec: A Spec object or dict containing the design specification
        output_path: Path where the rendered image will be saved
        canvas_width: Width of the canvas in pixels (default: 800)
        canvas_height: Height of the canvas in pixels (default: 600)

    Returns:
        Path: The output path where the image was saved
    """
    # Convert dict to Spec if needed
    if isinstance(spec, dict):
        spec = Spec(**spec)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate HTML
    html_content = _generate_html(spec, canvas_width, canvas_height)

    # Render with Playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': canvas_width, 'height': canvas_height})
        page.set_content(html_content)

        # Wait for any fonts to load
        page.wait_for_timeout(500)

        # Take screenshot
        page.screenshot(path=str(output_path), full_page=False)
        browser.close()

    return output_path