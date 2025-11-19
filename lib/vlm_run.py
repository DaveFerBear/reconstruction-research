from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI
from vlmrun.client import VLMRun

# Load API key
load_dotenv(Path(__file__).parent.parent / ".env")
VLMRUN_API_KEY = os.getenv("VLMRUN_API_KEY")
if not VLMRUN_API_KEY:
    raise RuntimeError("VLMRUN_API_KEY not set")

# Use VLM Run's OpenAI-compatible blocking API
chat = OpenAI(api_key=VLMRUN_API_KEY, base_url="https://agent.vlm.run/v1/openai")

# Upload local file to VLM Run object store for a public URL
uploader = VLMRun(api_key=VLMRUN_API_KEY, base_url="https://agent.vlm.run/v1")
uploaded = uploader.files.upload(file=Path("./lib/1600w-ijK0RlZ5JR8-1.webp"))
image_url = uploaded.public_url

# Blocking chat completion
resp = chat.chat.completions.create(
    model="vlmrun-orion-1:auto",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ],
    max_tokens=500,
)

# Print assistant message
print(resp.choices[0].message.content)