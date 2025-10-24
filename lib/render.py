from pathlib import Path
from typing import Union
import json
from playwright.sync_api import sync_playwright
from .types import Spec, TextNode, ImageNode


def _generate_html(spec: Spec, canvas_width: int = 800, canvas_height: int = 600, asset_dir: Path = None) -> str:
    """Generate HTML from a spec.

    Args:
        spec: The design spec
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        asset_dir: Directory containing generated asset images (asset-1.png, asset-2.png, etc.)
    """
    from .utils import _to_data_url

    # Collect unique fonts from the spec
    fonts_needed = set()
    for node in spec.nodes:
        if isinstance(node, TextNode):
            fonts_needed.add(node.font_family)

    # Map to Google Fonts (skip system fonts)
    google_fonts = {
        'Anton': 'family=Anton',
        'Dancing Script': 'family=Dancing+Script:wght@400;700',
        'Great Vibes': 'family=Great+Vibes',
        'Montserrat': 'family=Montserrat:wght@100;400;700;900',
        'Poppins': 'family=Poppins:wght@100;400;700;900',
    }

    font_imports = []
    for font in fonts_needed:
        if font in google_fonts:
            font_imports.append(google_fonts[font])

    # Build Google Fonts URL
    if font_imports:
        fonts_url = f"https://fonts.googleapis.com/css2?{'&'.join(font_imports)}&display=swap"
    else:
        fonts_url = None

    # Build node HTML
    nodes_html = []
    image_node_idx = 0
    for node in spec.nodes:
        if isinstance(node, TextNode):
            opacity = getattr(node, 'opacity', 1)
            style = (
                f"position: absolute; "
                f"left: {node.x}px; "
                f"top: {node.y}px; "
                f"width: {node.width}px; "
                f"height: {node.height}px; "
                f"transform: rotate({node.rotation}deg); "
                f"opacity: {opacity}; "
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
            image_node_idx += 1

            # Check if we have a generated asset for this image
            image_src = None
            if asset_dir:
                asset_path = asset_dir / f"asset-{image_node_idx}.png"
                if asset_path.exists():
                    image_src = _to_data_url(asset_path)

            if image_src:
                # Use actual image with object-fit to stretch to dimensions
                opacity = getattr(node, 'opacity', 1)
                style = (
                    f"position: absolute; "
                    f"left: {node.x}px; "
                    f"top: {node.y}px; "
                    f"width: {node.width}px; "
                    f"height: {node.height}px; "
                    f"transform: rotate({node.rotation}deg); "
                    f"opacity: {opacity}; "
                    f"object-fit: fill; "
                )
                nodes_html.append(f'<img src="{image_src}" style="{style}" alt="{node.asset_description}" />')
            else:
                # Show placeholder
                opacity = getattr(node, 'opacity', 1)
                style = (
                    f"position: absolute; "
                    f"left: {node.x}px; "
                    f"top: {node.y}px; "
                    f"width: {node.width}px; "
                    f"height: {node.height}px; "
                    f"transform: rotate({node.rotation}deg); "
                    f"opacity: {opacity}; "
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
                desc = node.asset_description[:100]
                nodes_html.append(f'<div style="{style}" title="{node.asset_description}">[Image: {desc}]</div>')

    # Build background style
    bg_style = f"background-color: {spec.background_color};"
    if spec.has_background_image and spec.background_image_description:
        # Check if we have a generated background image
        if asset_dir:
            bg_image_path = asset_dir / "background.png"
            if bg_image_path.exists():
                bg_data_url = _to_data_url(bg_image_path)
                bg_style = f"background-image: url('{bg_data_url}'); background-size: cover; background-position: center;"

    # Build font link tag if needed
    font_link = ''
    if fonts_url:
        font_link = f'''<link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="{fonts_url}" rel="stylesheet">'''

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    {font_link}
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
    asset_dir: Path = None,
) -> Path:
    """
    Render an image from a spec using a headless browser.

    Args:
        spec: A Spec object or dict containing the design specification
        output_path: Path where the rendered image will be saved
        canvas_width: Width of the canvas in pixels (default: 800)
        canvas_height: Height of the canvas in pixels (default: 600)
        asset_dir: Optional directory containing generated asset images

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
    html_content = _generate_html(spec, canvas_width, canvas_height, asset_dir=asset_dir)

    # Save HTML file alongside the image
    html_path = output_path.with_suffix('.html')
    html_path.write_text(html_content, encoding='utf-8')

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