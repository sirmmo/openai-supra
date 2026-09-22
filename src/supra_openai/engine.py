"""Load Supra2-IMG and sample images from text prompts.

Mirrors the reference sampler in the model repo's ``inference.py``: Flan-T5
text encoding, Euler integration of the rectified-flow ODE with
classifier-free guidance, then SD-VAE decoding.
"""

from __future__ import annotations

import logging
import random
import threading
import time

import torch
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from .model import LATENT_CH, LATENT_SIZE, PATCH, SupraDiT

log = logging.getLogger(__name__)

HF_REPO = "SupraLabs/Supra2-IMG"
CKPT_FILE = "model_final_ema.pt"
T5_NAME = "google/flan-t5-base"
VAE_NAME = "stabilityai/sd-vae-ft-mse"
VAE_SCALE = 0.18215
MAX_CTX_LEN = 128

IMAGE_SIZE = 256
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 3.0


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SupraEngine:
    """Holds the DiT, text encoder and VAE; ``generate`` is thread-safe."""

    def __init__(self, device: str | None = None, revision: str | None = None) -> None:
        self.device = pick_device(device)
        self._lock = threading.Lock()
        log.info("loading %s on %s", HF_REPO, self.device)
        t0 = time.perf_counter()

        path = hf_hub_download(HF_REPO, CKPT_FILE, revision=revision)
        state = torch.load(path, map_location="cpu", weights_only=True)
        cfg = state.get("config", {})
        if cfg.get("patch", PATCH) != PATCH:
            raise ValueError(f"checkpoint patch={cfg['patch']} != {PATCH}")
        weights = state.get("ema") or state.get("model") or state
        self.dit = SupraDiT()
        self.dit.load_state_dict(weights, strict=True)
        self.dit.to(self.device).eval()
        self.ctx_len = int(cfg.get("ctx_len", MAX_CTX_LEN))

        self.tokenizer = AutoTokenizer.from_pretrained(T5_NAME)
        self.text_encoder = T5EncoderModel.from_pretrained(T5_NAME).to(self.device).eval()
        self.vae = AutoencoderKL.from_pretrained(VAE_NAME, low_cpu_mem_usage=False).to(self.device).eval()

        if "uncond_text" in cfg:
            self.uncond_ctx = cfg["uncond_text"].float().unsqueeze(0).to(self.device)
            self.uncond_mask = cfg["uncond_mask"].float().unsqueeze(0).to(self.device)
        else:
            self.uncond_ctx, self.uncond_mask = self._encode([""])

        log.info("model ready in %.1fs", time.perf_counter() - t0)

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda")

    @torch.inference_mode()
    def _encode(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.ctx_len,
            return_tensors="pt",
        ).to(self.device)
        with self._autocast():
            ctx = self.text_encoder(**tok).last_hidden_state.float()
        return ctx, tok["attention_mask"].float()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        n: int = 1,
        *,
        seed: int | None = None,
        guidance_scale: float | None = None,
        steps: int | None = None,
    ) -> list[Image.Image]:
        """Return ``n`` 256x256 images for ``prompt``."""
        guidance_scale = DEFAULT_GUIDANCE if guidance_scale is None else guidance_scale
        steps = steps or DEFAULT_STEPS
        if seed is None:
            seed = random.randrange(2**63)

        with self._lock:
            t0 = time.perf_counter()
            ctx, mask = self._encode([prompt] * n)
            use_cfg = guidance_scale > 1.0
            if use_cfg:
                ctx = torch.cat([ctx, self.uncond_ctx.expand(n, -1, -1)])
                mask = torch.cat([mask, self.uncond_mask.expand(n, -1)])

            # Noise is drawn on the CPU so a seed gives the same image on any device.
            gen = torch.Generator().manual_seed(seed)
            z = torch.randn(n, LATENT_CH, LATENT_SIZE, LATENT_SIZE, generator=gen).to(self.device)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((n,), i * dt, device=self.device)
                with self._autocast():
                    if use_cfg:
                        v = self.dit(torch.cat([z, z]), torch.cat([t, t]), ctx, mask).float()
                        v_cond, v_uncond = v.chunk(2)
                        v = v_uncond + guidance_scale * (v_cond - v_uncond)
                    else:
                        v = self.dit(z, t, ctx, mask).float()
                z = z + dt * v

            with self._autocast():
                pixels = self.vae.decode(z / VAE_SCALE).sample.float()
            log.info(
                "generated n=%d steps=%d cfg=%.2f seed=%d in %.1fs",
                n, steps, guidance_scale, seed, time.perf_counter() - t0,
            )

        arr = ((pixels.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        arr = arr.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(a) for a in arr]
