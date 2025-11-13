"""Image captioning functionality using GPT-5 vision API and Moondream."""

import os
import base64
import tempfile
import requests
from pathlib import Path
import litellm
from dotenv import load_dotenv

load_dotenv()


def encode_image_to_data_url(image_path: Path) -> str:
    """Encode image to data URL for GPT-5 vision API."""
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
    }
    mime_type = mime_types.get(image_path.suffix.lower(), 'image/png')

    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    return f"data:{mime_type};base64,{image_data}"


def caption_image(
    image_path: Path,
    prompt: str = "Describe this image concisely in 1-2 sentences. Focus on the main subject, colors, style, and composition. Be specific and visual.",
    model: str = "gpt-5",
    timeout: int = 60
) -> str | None:
    """
    Generate a caption for an image using GPT-5 vision.

    Args:
        image_path: Path to the image file
        prompt: The prompt to use for captioning (default: concise visual description)
        model: Model to use (default: gpt-5)
        timeout: Request timeout in seconds (default: 60)

    Returns:
        str: The generated caption, or None if captioning fails
    """
    if not image_path.exists():
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    data_url = encode_image_to_data_url(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                }
            ]
        }
    ]

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
            timeout=timeout
        )

        caption = response.choices[0].message.content.strip()
        return caption
    except Exception as e:
        print(f"  Error captioning {image_path.name}: {e}")
        return None


def _render_svg_to_png(svg_path: Path, output_path: Path = None) -> Path:
    """
    Render an SVG file to PNG using Playwright.

    Args:
        svg_path: Path to the SVG file.
        output_path: Optional path for output PNG. If None, creates temp file.

    Returns:
        Path: Path to the rendered PNG file.
    """
    from playwright.sync_api import sync_playwright

    if output_path is None:
        # Create temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        output_path = Path(temp_path)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read SVG content
    svg_content = svg_path.read_text(encoding='utf-8')

    # Wrap SVG in HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100vw;
            height: 100vh;
            background: white;
        }}
        svg {{
            max-width: 100%;
            max-height: 100%;
        }}
    </style>
</head>
<body>
    {svg_content}
</body>
</html>"""

    # Create temporary HTML file
    fd, temp_html = tempfile.mkstemp(suffix='.html')
    os.close(fd)
    temp_html_path = Path(temp_html)
    temp_html_path.write_text(html, encoding='utf-8')

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': 1024, 'height': 1024})
            page.goto(f'file://{temp_html_path.resolve()}')
            page.wait_for_timeout(500)
            page.screenshot(path=str(output_path), full_page=False)
            browser.close()
    finally:
        # Clean up temp HTML
        temp_html_path.unlink()

    return output_path


def caption_moondream(
    image_path: Path,
    prompt: str = "Describe this image in detail.",
    timeout: int = 120
) -> str | None:
    """
    Generate a caption for an image using Moondream3 via FAL API.

    Args:
        image_path: Path to the image file (PNG, JPG, WEBP) or SVG file.
        prompt: The prompt/question to ask about the image.
        timeout: Request timeout in seconds (default: 120).

    Returns:
        str: The generated caption, or None if captioning fails.
    """
    if not image_path.exists():
        return None

    fal_api_key = os.getenv("FAL_API_KEY")
    if not fal_api_key:
        raise ValueError("FAL_API_KEY not set in environment")

    image_path = Path(image_path)
    temp_png = None

    try:
        # If SVG, render it first
        if image_path.suffix.lower() == '.svg':
            temp_png = _render_svg_to_png(image_path)
            image_path = temp_png

        # Encode image to data URL
        data_url = encode_image_to_data_url(image_path)

        # Call Moondream3 via FAL
        headers = {"Authorization": f"Key {fal_api_key}"}
        payload = {
            "image_url": data_url,
            "prompt": prompt
        }

        response = requests.post(
            "https://fal.run/fal-ai/moondream3-preview/caption",
            json=payload,
            headers=headers,
            timeout=timeout
        )

        response.raise_for_status()

        if response.status_code == 200:
            result = response.json()
        elif response.status_code == 202:
            job = response.json()
            status_url = job.get("status_url") or job.get("response_url")

            # Poll for completion
            import time
            while True:
                time.sleep(2)
                r = requests.get(status_url, headers=headers, timeout=timeout)
                r.raise_for_status()
                result = r.json()
                state = (result.get("status") or result.get("state") or "").lower()

                if state in ("completed", "success", "succeeded"):
                    break
                if state in ("failed", "error"):
                    raise RuntimeError(f"FAL job failed: {result}")
        else:
            raise RuntimeError(f"Unexpected response: {response.status_code}")

        # Extract caption from result
        if "output" in result:
            return result["output"]
        elif "caption" in result:
            return result["caption"]
        else:
            raise ValueError(f"Could not extract caption from response: {result}")

    except Exception as e:
        print(f"  Error captioning {image_path.name}: {e}")
        return None
    finally:
        # Clean up temporary PNG if created
        if temp_png and temp_png.exists():
            temp_png.unlink()
