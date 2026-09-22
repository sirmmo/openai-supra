# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
# CPU wheels by default. For NVIDIA GPUs build with e.g.
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/data/hf
RUN pip install uv && uv pip install torch --index-url "$TORCH_INDEX_URL"
WORKDIR /app
# Dependencies first so code edits don't invalidate this layer.
COPY pyproject.toml ./
RUN uv pip install -r pyproject.toml
COPY README.md ./
COPY src ./src
RUN uv pip install --no-deps .

FROM base AS test
RUN uv pip install -r pyproject.toml --extra dev
COPY tests ./tests
COPY examples ./examples
CMD ["pytest", "-q"]

FROM base AS runtime
RUN useradd --create-home --uid 1000 app && mkdir -p /data/hf && chown app /data/hf
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "supra_openai.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
