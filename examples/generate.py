"""Generate an image from a running supra-openai server with the OpenAI SDK.

    python examples/generate.py "a lighthouse on a cliff at sunset" out.png
"""

import base64
import os
import sys

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
)

prompt = sys.argv[1] if len(sys.argv) > 1 else "a sea jellyfish floating in the pitch-black ocean depths"
out = sys.argv[2] if len(sys.argv) > 2 else "out.png"

result = client.images.generate(
    model="supra2-img",
    prompt=prompt,
    size="256x256",
    # Supra-specific knobs (optional): reference settings from the model card.
    extra_body={"seed": 0, "guidance_scale": 3.0, "steps": 50},
)
with open(out, "wb") as f:
    f.write(base64.b64decode(result.data[0].b64_json))
print(f"saved {out}")
