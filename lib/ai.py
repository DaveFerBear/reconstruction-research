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


def flux_generate(prompt: str, timeout: int = 120) -> dict:
    """
    Generate an image using FLUX.1 Pro text-to-image model.

    Args:
        prompt: The image description to generate.
        timeout: Max request time in seconds.

    Returns:
        dict: JSON result from FAL API containing the generated image.
    """

    headers = {"Authorization": f"Key {FAL_API_KEY}"}
    payload = {"prompt": prompt}

    response = requests.post("https://fal.run/fal-ai/flux-pro/v1.1", json=payload, headers=headers, timeout=timeout)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Print the response body for debugging
        print(f"API Error Response: {response.text}")
        raise e

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


def remove_background(image_url: str, timeout: int = 120) -> dict:
    """
    Remove background from an image to create transparency.

    Args:
        image_url: URL of the image to process.
        timeout: Max request time in seconds.

    Returns:
        dict: JSON result from FAL API containing the image with removed background.
    """

    headers = {"Authorization": f"Key {FAL_API_KEY}"}
    payload = {"image_url": image_url}

    response = requests.post("https://fal.run/fal-ai/imageutils/rembg", json=payload, headers=headers, timeout=timeout)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"API Error Response: {response.text}")
        raise e

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


def kontext_edit(prompt: str, image_url: str, with_logs: bool = True, timeout: int = 120) -> dict:
    """
    Context-aware image editing using FAL flux-pro/kontext model.

    Args:
        prompt: The edit instruction, e.g. "Put a donut next to the flour."
        image_url: URL of the image to edit.
        with_logs: Whether to print progress logs.
        timeout: Max request time in seconds.

    Returns:
        dict: JSON result from FAL API containing the edited image.
    """

    headers = {"Authorization": f"Key {FAL_API_KEY}"}
    payload = {"prompt": prompt, "image_url": image_url}

    response = requests.post("https://fal.run/fal-ai/flux-pro/kontext", json=payload, headers=headers, timeout=timeout)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Print the response body for debugging
        print(f"API Error Response: {response.text}")
        raise e

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

            if with_logs and "logs" in data:
                for log in data["logs"]:
                    print(log.get("message", ""))

            if state in ("completed", "success", "succeeded"):
                return data
            if state in ("failed", "error"):
                raise RuntimeError(f"FAL job failed: {data}")
    else:
        raise RuntimeError(f"Unexpected response: {response.status_code} - {response.text}")
