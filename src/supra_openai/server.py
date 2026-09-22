"""OpenAI-compatible HTTP API for Supra2-IMG.

Point the official OpenAI SDK at this server and call ``client.images.generate``::

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
    client.images.generate(model="supra2-img", prompt="a red fox in snow")

Configuration (environment):
    SUPRA_API_KEY     require ``Authorization: Bearer <key>`` when set
    SUPRA_DEVICE      torch device (cpu, cuda, cuda:1, mps); auto-detected when unset
    SUPRA_REVISION    Hugging Face revision of SupraLabs/Supra2-IMG to load
    SUPRA_MODEL_NAME  model id reported by /v1/models (default: supra2-img)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import secrets
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Protocol

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

NATIVE_SIZE = 256
MIN_SIDE, MAX_SIDE = 64, 2048
# OpenAI's `quality` picks a step count; anything else uses the engine default.
QUALITY_STEPS = {"low": 20, "medium": 35}
URL_CACHE_SIZE = 100
MEDIA_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
_SIZE_RE = re.compile(r"(\d+)x(\d+)")
UI_PAGE = Path(__file__).parent / "static" / "index.html"


class ImageEngine(Protocol):
    def generate(
        self,
        prompt: str,
        n: int = 1,
        *,
        seed: int | None = None,
        guidance_scale: float | None = None,
        steps: int | None = None,
    ) -> list[Image.Image]: ...


class ImageGenerationRequest(BaseModel):
    """Body of ``POST /v1/images/generations``.

    Other OpenAI fields (style, user, background, moderation, ...) are accepted
    and ignored. ``seed``, ``guidance_scale`` and ``steps`` are Supra extensions;
    send them with the SDK's ``extra_body``.
    """

    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1)
    model: str | None = None
    n: int = Field(1, ge=1, le=10)
    size: str | None = None
    quality: str | None = None
    response_format: Literal["b64_json", "url"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None
    output_compression: int | None = Field(None, ge=0, le=100)
    stream: bool | None = None

    seed: int | None = Field(None, ge=0, lt=2**63)
    guidance_scale: float | None = Field(
        None, ge=0, le=20, validation_alias=AliasChoices("guidance_scale", "cfg")
    )
    steps: int | None = Field(
        None, ge=1, le=200, validation_alias=AliasChoices("steps", "num_inference_steps")
    )


class APIError(Exception):
    def __init__(self, status: int, message: str, param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.status, self.message, self.param, self.code = status, message, param, code


def _error(status: int, message: str, param: str | None = None, code: str | None = None) -> JSONResponse:
    kind = "server_error" if status >= 500 else "invalid_request_error"
    body = {"error": {"message": message, "type": kind, "param": param, "code": code}}
    return JSONResponse(body, status_code=status)


def _parse_size(size: str | None) -> tuple[int, int]:
    if size in (None, "auto"):
        return NATIVE_SIZE, NATIVE_SIZE
    m = _SIZE_RE.fullmatch(size)
    if not m or not all(MIN_SIDE <= int(side) <= MAX_SIDE for side in m.groups()):
        raise APIError(
            400,
            f"Invalid size '{size}'. Use 'WIDTHxHEIGHT' with sides between {MIN_SIDE} and "
            f"{MAX_SIDE}, or 'auto' for the native {NATIVE_SIZE}x{NATIVE_SIZE}.",
            param="size",
        )
    return int(m[1]), int(m[2])


def _encode(img: Image.Image, size: tuple[int, int], fmt: str, compression: int | None) -> bytes:
    if img.size != size:
        # The model only renders 256x256: scale to cover, then center-crop.
        img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, "PNG")
    else:
        img.save(buf, fmt.upper(), quality=90 if compression is None else compression)
    return buf.getvalue()


class _ImageStore:
    """Most recent generated images, served for ``response_format="url"``."""

    def __init__(self, capacity: int) -> None:
        self._items: OrderedDict[str, bytes] = OrderedDict()
        self._capacity = capacity

    def put(self, name: str, blob: bytes) -> None:
        self._items[name] = blob
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)

    def get(self, name: str) -> bytes | None:
        return self._items.get(name)


def create_app(engine: ImageEngine | None = None, *, api_key: str | None = None) -> FastAPI:
    """Build the app. Without ``engine``, Supra2-IMG is loaded at startup."""
    api_key = api_key or os.environ.get("SUPRA_API_KEY") or None
    model_id = os.environ.get("SUPRA_MODEL_NAME", "supra2-img")
    started = int(time.time())
    store = _ImageStore(URL_CACHE_SIZE)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            logging.basicConfig(format="%(levelname)s:     %(name)s - %(message)s")
            logging.getLogger("supra_openai").setLevel(logging.INFO)
            from .engine import SupraEngine

            app.state.engine = await run_in_threadpool(
                SupraEngine,
                device=os.environ.get("SUPRA_DEVICE") or None,
                revision=os.environ.get("SUPRA_REVISION") or None,
            )
        yield

    app = FastAPI(title="Supra2-IMG OpenAI-compatible API", lifespan=lifespan)
    app.state.engine = engine

    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError):
        return _error(exc.status, exc.message, exc.param, exc.code)

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        return _error(exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        err = exc.errors()[0]
        param = ".".join(str(p) for p in err["loc"] if p != "body") or None
        return _error(400, f"{param}: {err['msg']}" if param else err["msg"], param=param)

    def check_auth(request: Request) -> None:
        if api_key is None:
            return
        supplied = request.headers.get("authorization", "")
        if not secrets.compare_digest(supplied.encode(), f"Bearer {api_key}".encode()):
            raise APIError(401, "Incorrect API key provided.", code="invalid_api_key")

    model_card = {"id": model_id, "object": "model", "created": started, "owned_by": "SupraLabs"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    async def ui():
        return FileResponse(UI_PAGE, media_type="text/html")

    # /v1 is only a base-URL prefix for clients; answer with a pointer instead of a 404.
    @app.get("/v1")
    async def index():
        return {
            "service": "supra-openai",
            "model": model_id,
            "base_url": "/v1",
            "auth_required": api_key is not None,
            "endpoints": ["POST /v1/images/generations", "GET /v1/models", "GET /health"],
            "docs": "/docs",
        }

    @app.get("/v1/models", dependencies=[Depends(check_auth)])
    async def list_models():
        return {"object": "list", "data": [model_card]}

    @app.get("/v1/models/{name:path}", dependencies=[Depends(check_auth)])
    async def get_model(name: str):
        if name != model_id:
            raise APIError(404, f"The model '{name}' does not exist.", param="model", code="model_not_found")
        return model_card

    @app.post("/v1/images/generations", dependencies=[Depends(check_auth)])
    async def create_image(body: ImageGenerationRequest, request: Request):
        if body.stream:
            raise APIError(400, "Streaming is not supported by this server.", param="stream")
        size = _parse_size(body.size)
        fmt = body.output_format or "png"
        images = await run_in_threadpool(
            request.app.state.engine.generate,
            body.prompt,
            body.n,
            seed=body.seed,
            guidance_scale=body.guidance_scale,
            steps=body.steps or QUALITY_STEPS.get(body.quality or ""),
        )

        data = []
        for img in images:
            blob = _encode(img, size, fmt, body.output_compression)
            if body.response_format == "url":
                name = f"{uuid.uuid4().hex}.{fmt}"
                store.put(name, blob)
                data.append({"url": str(request.url_for("get_image_file", name=name))})
            else:
                data.append({"b64_json": base64.b64encode(blob).decode()})
        return {"created": int(time.time()), "data": data, "output_format": fmt, "size": f"{size[0]}x{size[1]}"}

    # Unauthenticated like OpenAI's signed URLs; names are unguessable UUIDs.
    @app.get("/v1/images/files/{name}", name="get_image_file")
    async def get_image_file(name: str):
        blob = store.get(name)
        if blob is None:
            raise APIError(404, "Image not found or expired.")
        return Response(blob, media_type=MEDIA_TYPES[name.rsplit(".", 1)[-1]])

    return app
