import os
import requests
from dotenv import load_dotenv

load_dotenv()

FAL_API_KEY = os.getenv("FAL_API_KEY")
FAL_EDIT_URL = "https://fal.run/fal-ai/nano-banana/edit"


def edit_image(prompt: str, image_urls: list[str], timeout: int = 120) -> dict:
    """
    Stateless image edit using FAL nano-banana model.

    Args:
        prompt: The edit instruction, e.g. "Extract the man from this graphic design."
        image_urls: A list of image URLs to edit.
        timeout: Max request time in seconds.

    Returns:
        dict: JSON result from FAL API (may include output image URLs, metadata, etc.)
    """

    headers = {"Authorization": f"Key {FAL_API_KEY}"}
    payload = {"prompt": prompt, "image_urls": image_urls}

    response = requests.post(FAL_EDIT_URL, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 202:
        job = response.json()
        status_url = job.get("status_url") or job.get("response_url")

        while True:
            r = requests.get(status_url, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            state = (data.get("status") or data.get("state") or "").lower()
            if state in ("completed", "success", "succeeded"):
                return data
            if state in ("failed", "error"):
                raise RuntimeError(f"FAL job failed: {data}")
    else:
        raise RuntimeError(f"Unexpected response: {response.status_code} - {response.text}")
