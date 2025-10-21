import base64
import json
import re
from pathlib import Path


MIME_BY_SUFFIX = {
    '.webp': 'image/webp',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
}


def _to_data_url(image_path: Path) -> str:
    mime = MIME_BY_SUFFIX.get(image_path.suffix.lower(), 'application/octet-stream')
    b64 = base64.b64encode(image_path.read_bytes()).decode('utf-8')
    return f'data:{mime};base64,{b64}'


_json_block_re = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


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
