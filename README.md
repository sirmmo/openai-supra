# supra-openai

[![CI](https://github.com/sirmmo/openai-supra/actions/workflows/ci.yml/badge.svg)](https://github.com/sirmmo/openai-supra/actions/workflows/ci.yml)
[![Publish image](https://github.com/sirmmo/openai-supra/actions/workflows/publish.yml/badge.svg)](https://github.com/sirmmo/openai-supra/actions/workflows/publish.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

An OpenAI-compatible Images API server for
[SupraLabs/Supra2-IMG](https://huggingface.co/SupraLabs/Supra2-IMG), a ~100M-parameter
text-to-image DiT. It exposes `POST /v1/images/generations`, so the official `openai`
SDKs (Python, JS, ...) and other OpenAI-compatible tools work by changing `base_url`.

## Run with Docker

Published images are at `ghcr.io/sirmmo/openai-supra` (linux/amd64 and linux/arm64):

```bash
docker run -p 8000:8000 -v supra-hf:/data/hf ghcr.io/sirmmo/openai-supra:latest
```

Or build from source with compose:

```bash
docker compose up -d --build        # serves on http://localhost:8000
docker compose logs -f supra        # first start downloads ~1.8 GB of weights
```

Tags: `latest` and `main` track the default branch, `0.1.0` / `0.1` come from release
tags, and `sha-<commit>` pins an exact build. Images carry build provenance and a cosign
signature; verify one with:

```bash
cosign verify ghcr.io/sirmmo/openai-supra:latest \
  --certificate-identity-regexp '^https://github.com/sirmmo/openai-supra/' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
gh attestation verify oci://ghcr.io/sirmmo/openai-supra:latest --repo sirmmo/openai-supra
```

Weights (the Supra checkpoint, `google/flan-t5-base`, `stabilityai/sd-vae-ft-mse`) are
cached in the `hf-cache` volume, so later starts only need to load them.

Pick the Python version with a build arg (default `3.12`, needs `>=3.10`):

```bash
docker build --build-arg PYTHON_VERSION=3.11 -t supra-openai .
docker run -p 8000:8000 -v supra-hf:/data/hf supra-openai
```

The image uses CPU PyTorch by default. For an NVIDIA GPU, build CUDA wheels and run with
the NVIDIA container toolkit:

```bash
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 -t supra-openai:cuda .
docker run --gpus all -p 8000:8000 -v supra-hf:/data/hf supra-openai:cuda
```

### Configuration

| Variable           | Default      | Meaning                                                      |
| ------------------ | ------------ | ------------------------------------------------------------ |
| `SUPRA_API_KEY`    | unset        | If set, require `Authorization: Bearer <key>`                |
| `SUPRA_DEVICE`     | auto         | `cpu`, `cuda`, `cuda:1`, `mps`                               |
| `SUPRA_REVISION`   | `main`       | Hugging Face revision (commit/tag) of the Supra checkpoint   |
| `SUPRA_MODEL_NAME` | `supra2-img` | Model id reported by `/v1/models`                            |
| `SUPRA_PORT`       | `8000`       | Host port (compose only)                                     |
| `SUPRA_BIND`       | `0.0.0.0`    | Host interface to publish on; `127.0.0.1` keeps it local (compose only) |

## Built-in web form

Open the server's root (e.g. <http://localhost:8000/>) for a small generation form:
prompt, image count, size, steps, guidance and seed, with the seed shown under each
result so you can reproduce or download it. It is served by the same app and calls the
same `/v1/images/generations` endpoint, so nothing else needs to run. When
`SUPRA_API_KEY` is set the page asks for the key and keeps it for the browser session.

## Use it from the OpenAI SDK

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

result = client.images.generate(
    model="supra2-img",
    prompt="a sea jellyfish floating in the pitch-black ocean depths",
    size="256x256",
    extra_body={"seed": 0, "guidance_scale": 3.0, "steps": 50},  # optional
)
open("jellyfish.png", "wb").write(base64.b64decode(result.data[0].b64_json))
```

Or run `python examples/generate.py "a lighthouse at dusk" out.png`
(it honours `OPENAI_BASE_URL` / `OPENAI_API_KEY`).

With curl:

```bash
curl http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox in the snow", "n": 2, "seed": 1}' \
  | jq -r '.data[0].b64_json' | base64 -d > fox.png
```

## API

`POST /v1/images/generations`

| Field                | Supported values                                                                       |
| -------------------- | -------------------------------------------------------------------------------------- |
| `prompt`             | required; Flan-T5 sees the first 128 tokens                                            |
| `model`              | accepted and ignored (single-model server)                                             |
| `n`                  | 1–10, generated as one batch                                                           |
| `size`               | `auto` or `256x256` (native). Other `WxH` (64–2048) are upscaled and center-cropped   |
| `quality`            | `low` = 20 steps, `medium` = 35; anything else uses 50                                 |
| `response_format`    | `b64_json` (default) or `url`                                                          |
| `output_format`      | `png` (default), `jpeg`, `webp`; `output_compression` 0–100 for jpeg/webp              |
| `stream`             | not supported (400)                                                                    |
| `style`, `user`, ... | accepted and ignored                                                                   |

Supra extensions (pass via `extra_body` in the SDK):

| Field            | Alias                 | Default | Notes                                 |
| ---------------- | --------------------- | ------- | ------------------------------------- |
| `seed`           |                       | random  | Same seed gives the same image on any device |
| `guidance_scale` | `cfg`                 | 3.0     | Classifier-free guidance; ≤1 disables |
| `steps`          | `num_inference_steps` | 50      | Euler steps, 1–200                    |

`url` responses point at `/v1/images/files/<id>`; the last 100 images are kept in memory.

Other endpoints: `GET /` (web form), `GET /v1` (service info), `GET /v1/models`,
`GET /v1/models/{id}`, `GET /docs` (OpenAPI), `GET /health`.
Errors use OpenAI's `{"error": {"message", "type", "param", "code"}}` envelope, so the
SDK raises the usual `BadRequestError`, `AuthenticationError`, etc.

The model only renders 256×256. Larger sizes are resampled, not generated at higher
resolution. Generation runs one request at a time; on CPU, 50 steps take roughly 10–15 s
(13 s measured on a 40-core Xeon).

## Tests

```bash
docker build --target test -t supra-openai:test . && docker run --rm supra-openai:test
```

The tests drive the API through the real `openai` SDK against a fake engine, so they
don't download weights.

## Layout

- `src/supra_openai/model.py`: SupraDiT architecture, vendored from the model repo's `inference.py` (Apache-2.0)
- `src/supra_openai/engine.py`: loads the checkpoint, Flan-T5 and the VAE, and runs the rectified-flow sampler
- `src/supra_openai/server.py`: FastAPI app with the OpenAI-compatible routes
- `src/supra_openai/static/index.html`: the built-in web form (no external assets)
- `.github/workflows/`: `ci.yml` (tests on Python 3.10–3.13), `publish.yml` (multi-arch
  GHCR image, provenance, cosign signature, Trivy scan, release), `codeql.yml`
