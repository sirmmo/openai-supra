"""Exercise the API through the official OpenAI SDK against a fake engine."""

import base64
import io

import openai
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from supra_openai.server import create_app


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, n=1, *, seed=None, guidance_scale=None, steps=None):
        self.calls.append(dict(prompt=prompt, n=n, seed=seed, guidance_scale=guidance_scale, steps=steps))
        return [Image.new("RGB", (256, 256), (200, 30, 30)) for _ in range(n)]


def sdk(app, api_key="unused"):
    return openai.OpenAI(base_url="http://testserver/v1", api_key=api_key, http_client=TestClient(app))


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(engine):
    return sdk(create_app(engine))


def decode(item):
    return Image.open(io.BytesIO(base64.b64decode(item.b64_json)))


def test_generate_defaults_to_native_png(client, engine):
    res = client.images.generate(model="supra2-img", prompt="a cat")

    assert len(res.data) == 1
    img = decode(res.data[0])
    assert (img.format, img.size) == ("PNG", (256, 256))
    assert res.size == "256x256"
    assert engine.calls == [dict(prompt="a cat", n=1, seed=None, guidance_scale=None, steps=None)]


def test_size_format_and_extensions(client, engine):
    res = client.images.generate(
        model="supra2-img",
        prompt="a dog",
        n=2,
        size="1024x768",
        output_format="jpeg",
        extra_body={"seed": 7, "guidance_scale": 4.5, "steps": 12},
    )

    assert [decode(d).size for d in res.data] == [(1024, 768)] * 2
    assert decode(res.data[0]).format == "JPEG"
    assert engine.calls[0] == dict(prompt="a dog", n=2, seed=7, guidance_scale=4.5, steps=12)


def test_quality_selects_steps_and_aliases_work(client, engine):
    client.images.generate(model="supra2-img", prompt="x", quality="low")
    client.images.generate(model="supra2-img", prompt="x", quality="low", extra_body={"num_inference_steps": 5, "cfg": 2})

    assert engine.calls[0]["steps"] == 20
    assert (engine.calls[1]["steps"], engine.calls[1]["guidance_scale"]) == (5, 2)


def test_url_response_is_fetchable(engine):
    app = create_app(engine)
    res = sdk(app).images.generate(model="supra2-img", prompt="a boat", response_format="url")

    fetched = TestClient(app).get(res.data[0].url)
    assert fetched.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(fetched.content)).size == (256, 256)


@pytest.mark.parametrize(
    "kwargs, param",
    [
        (dict(size="10x10"), "size"),
        (dict(size="big"), "size"),
        (dict(n=11), "n"),
        (dict(extra_body={"steps": 0}), "steps"),
        (dict(stream=True), "stream"),
    ],
)
def test_invalid_requests_return_openai_errors(client, engine, kwargs, param):
    with pytest.raises(openai.BadRequestError) as exc:
        client.images.generate(model="supra2-img", prompt="x", **kwargs)

    assert exc.value.param == param
    assert exc.value.type == "invalid_request_error"
    assert engine.calls == []


def test_api_key(engine):
    app = create_app(engine, api_key="secret")

    with pytest.raises(openai.AuthenticationError):
        sdk(app, api_key="wrong").images.generate(model="supra2-img", prompt="x")
    assert sdk(app, api_key="secret").images.generate(model="supra2-img", prompt="x").data


def test_ui_and_index(engine):
    http = TestClient(create_app(engine))

    page = http.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "/v1/images/generations" in page.text

    index = http.get("/v1").json()
    assert (index["base_url"], index["auth_required"]) == ("/v1", False)
    assert TestClient(create_app(engine, api_key="k")).get("/v1").json()["auth_required"] is True


def test_ui_request_shape_is_accepted(client, engine):
    """The page posts its knobs as top-level JSON fields, like this."""
    body = client.post(
        "/images/generations",
        cast_to=dict,
        body={"model": "supra2-img", "prompt": "a fox", "n": 1, "size": "512x512",
              "seed": 12, "steps": 30, "guidance_scale": 3.5},
    )

    assert body["size"] == "512x512"
    assert engine.calls[0] == dict(prompt="a fox", n=1, seed=12, guidance_scale=3.5, steps=30)


def test_models(client):
    assert [m.id for m in client.models.list()] == ["supra2-img"]
    assert client.models.retrieve("supra2-img").owned_by == "SupraLabs"
    with pytest.raises(openai.NotFoundError):
        client.models.retrieve("dall-e-3")
