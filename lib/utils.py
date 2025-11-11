import base64
import json
import re
from pathlib import Path
from PIL import Image
from io import BytesIO


MIME_BY_SUFFIX = {
    '.webp': 'image/webp',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
}


def _resize_image(image_path: Path, max_size: int = 1024, quality: int = 85) -> bytes:
    """
    Resize an image to fit within max_size while maintaining aspect ratio.

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) in pixels
        quality: JPEG quality (1-100)

    Returns:
        bytes: JPEG-encoded image data
    """
    with Image.open(image_path) as im:
        # Convert to RGB if needed
        if im.mode in ('RGBA', 'LA', 'P'):
            im = im.convert('RGB')

        # Resize if larger than max_size
        if max(im.size) > max_size:
            im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Save to bytes
        buf = BytesIO()
        im.save(buf, format='JPEG', quality=quality, optimize=True)
        return buf.getvalue()


def _to_data_url(image_path: Path, max_size: int = None) -> str:
    """
    Convert an image to a base64 data URL.

    Args:
        image_path: Path to the image file
        max_size: Optional maximum dimension for resizing (defaults to no resizing)

    Returns:
        str: Data URL (data:image/...;base64,...)
    """
    if max_size is not None:
        # Resize and convert to JPEG
        data = _resize_image(image_path, max_size=max_size)
        mime = 'image/jpeg'
    else:
        # Use original file
        data = image_path.read_bytes()
        mime = MIME_BY_SUFFIX.get(image_path.suffix.lower(), 'application/octet-stream')

    b64 = base64.b64encode(data).decode('utf-8')
    return f'data:{mime};base64,{b64}'


_json_block_re = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _load_rgb(path: Path) -> Image.Image:
    """
    Load an image and convert to RGB mode.

    Args:
        path: Path to the image file

    Returns:
        Image.Image: PIL Image in RGB mode
    """
    with Image.open(path) as im:
        return im.convert('RGB')


def image_to_data_url(image_path: Path, max_width: int = 360) -> str:
    """
    Convert an image to a base64 data URL with optional width-based resizing.

    Args:
        image_path: Path to the image file
        max_width: Maximum width in pixels (default 360)

    Returns:
        str: Data URL (data:image/png;base64,...)
    """
    with Image.open(image_path) as im:
        if im.width > max_width:
            ratio = max_width / im.width
            new_height = int(im.height * ratio)
            im = im.resize((max_width, new_height), Image.Resampling.LANCZOS)
        if im.mode in ('RGBA', 'LA', 'P'):
            im = im.convert('RGB')
        buf = BytesIO()
        im.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f'data:image/png;base64,{b64}'


def _parse_json_str(text: str):
    """Parse JSON from LLM response, handling code fences and extra text."""
    # First try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find JSON in markdown code fences
    m = _json_block_re.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Try to find JSON object in the text (look for {...})
    try:
        # Find first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
            return json.loads(json_str)
    except Exception:
        pass

    # If all else fails, print the response for debugging
    print(f"\n!!! Failed to parse JSON. Raw response:\n{text[:500]}...\n")
    raise ValueError('Model did not return valid JSON')
