"""Image captioning functionality using GPT-5 vision API."""

import os
import base64
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
