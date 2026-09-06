#!/usr/bin/env python
"""Photoreal fine-tune of the recovered neural renderer (roadmap item 6).

Self-supervised soft-to-sharp pairs: a sharp still is the target, its degraded copy
(downscale + resample + JPEG + slight blur, mimicking soft AI-video output) is the input.
The recovered graph is trained as-is: every E4M3 rounding gets a straight-through estimator,
the logical weights become trainable (attention bias/scale constants stay frozen). The loss is
composed the way inference composes: image + 0.25 * head[..., :3], L1 plus a high-pass L1.

  finetune.py build  --out ft/dataset.txt DIR [DIR ...]      # sharp stills with a face (skin >= 5%)
  finetune.py train  --dataset ft/dataset.txt --weights W --out ft/run1 [--steps 1500 --batch 2 --crop 256 --lr 2e-5]
  finetune.py eval   --dataset ft/dataset.txt --weights W [--weights2 FT] [--n 24]
"""
from __future__ import annotations

import argparse
import io
import math
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MLXDLSS_TORCH_CHUNK_TOKENS", "0")   # no in-place chunk writes under autograd

import numpy as np
import torch
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from mlxdlss import model as M                                   # noqa: E402
from mlxdlss.automask import SkinMasker                          # noqa: E402
from mlxdlss.features import PROFILES, NetworkGeometry, make_features   # noqa: E402
from mlxdlss.pipeline import NeuralRenderingPipeline             # noqa: E402

LOGICAL_FORMAT = {"format": "dlssnr-logical-v18", "fully_logical": "true"}


# ----------------------------------------------------------------------------- data

def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255


def _x264_roundtrip(rgb: np.ndarray, crf: int, rng: random.Random) -> np.ndarray:
    """One frame through libx264 at ``crf`` (yuv420p, 4:2:0 chroma) and back — the codec AI video
    actually ships in. A second pass (30%) mimics re-encoded uploads."""
    h, w = rgb.shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    data = np.ascontiguousarray(rgb[:h2, :w2]).tobytes()
    passes = 2 if rng.random() < 0.3 else 1
    for _ in range(passes):
        proc = subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}", "-i", "-",
                               "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-f", "h264", "-"],
                              input=data, capture_output=True, check=True)
        proc = subprocess.run(["ffmpeg", "-v", "error", "-f", "h264", "-i", "-", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                              input=proc.stdout, capture_output=True, check=True)
        data = proc.stdout[: w2 * h2 * 3]
        crf = max(18, crf - 6)
    out = rgb.copy()
    out[:h2, :w2] = np.frombuffer(data, np.uint8).reshape(h2, w2, 3)
    return out


def degrade(image: np.ndarray, rng: random.Random, version: int = 2) -> np.ndarray:
    """Soft-video proxy.

    v1 (RealSR-calibrated JPEG chain, kept 0.60 of the sharp high-pass): downscale 0.35-0.65,
    resample back, JPEG q25-55, blur σ0.5-1.3.
    v2 (rival panel 2026-09-06): mixture — 55% x264 round trip at CRF 30-40 / 4:2:0 (what AI video
    ships in), 20% the v1 JPEG chain, 10% "phone" (sensor noise + light blur, no codec), 15% clean
    (tiny blur only) so the network learns to leave sharp input alone. Sensor noise before the
    codec in half the samples; codec BEFORE blur; wider downscale/blur ranges."""
    h, w = image.shape[:2]
    if version == 1:
        pil = Image.fromarray((image * 255 + 0.5).astype(np.uint8))
        f = rng.uniform(0.35, 0.65)
        pil = pil.resize((max(64, int(w * f)), max(64, int(h * f))), Image.LANCZOS).resize((w, h), rng.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]))
        buf = io.BytesIO(); pil.save(buf, "JPEG", quality=rng.randint(25, 55)); buf.seek(0)
        pil = Image.open(buf).convert("RGB").filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.3)))
        return np.asarray(pil).astype(np.float32) / 255
    branch = rng.random()
    pil = Image.fromarray((image * 255 + 0.5).astype(np.uint8))
    if branch < 0.20:                                       # clean: already-sharp input, do (almost) nothing
        pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 0.4)))
        return np.asarray(pil).astype(np.float32) / 255
    if branch < 0.30:                                       # phone: noise + mild softness, no codec
        arr = np.asarray(pil).astype(np.float32)
        arr = arr + rng.uniform(1.0, 3.0) * np.random.default_rng(rng.randrange(1 << 30)).standard_normal(arr.shape).astype(np.float32)
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.0)))
        return np.asarray(pil).astype(np.float32) / 255
    f = rng.uniform(0.55, 1.0)                              # real H3 frames: skin hp p10 1.26 / mean 2.28 / p90 3.75 vs sharp 3.05 — a WIDE range up to near-sharp
    pil = pil.resize((max(64, int(w * f)), max(64, int(h * f))), Image.LANCZOS).resize((w, h), rng.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]))
    arr = np.asarray(pil).astype(np.float32)
    if rng.random() < 0.5:                                  # sensor noise before the codec
        arr = arr + rng.uniform(0.5, 2.5) * np.random.default_rng(rng.randrange(1 << 30)).standard_normal(arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if branch < 0.50:                                       # JPEG chain (v1-like, wider)
        buf = io.BytesIO(); Image.fromarray(arr).save(buf, "JPEG", quality=rng.randint(35, 75)); buf.seek(0)
        arr = np.asarray(Image.open(buf).convert("RGB"))
    else:                                                   # x264 round trip
        arr = _x264_roundtrip(arr, rng.randint(22, 36), rng)
    pil = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 0.7)))
    return np.asarray(pil).astype(np.float32) / 255


class Pairs:
    """Random skin-biased crops of (degraded, sharp, skin mask) from a list of sharp stills."""

    def __init__(self, paths: list[Path], crop: int, masker: SkinMasker, seed: int = 0, cache: int = 48, degrade_version: int = 2, mask_on_degraded: bool = True):
        self.paths, self.crop, self.masker, self.rng = paths, crop, masker, random.Random(seed)
        self.degrade_version, self.mask_on_degraded = degrade_version, mask_on_degraded
        self.cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        self.cache_limit = cache

    def still(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        if path not in self.cache:
            if len(self.cache) >= self.cache_limit:
                self.cache.pop(next(iter(self.cache)))
            sharp = load_image(path)
            self.cache[path] = (sharp, self.masker.skin(sharp))
        return self.cache[path]

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sharp, skin = self.still(self.rng.choice(self.paths))
        h, w = sharp.shape[:2]; c = self.crop
        if self.rng.random() < 0.7 and skin.max() > 0.5:
            ys, xs = np.nonzero(skin > 0.5); i = self.rng.randrange(len(ys))
            y0 = int(np.clip(ys[i] - c // 2, 0, h - c)); x0 = int(np.clip(xs[i] - c // 2, 0, w - c))
        else:
            y0 = self.rng.randrange(0, h - c + 1); x0 = self.rng.randrange(0, w - c + 1)
        tgt = sharp[y0:y0 + c, x0:x0 + c]
        soft = degrade(tgt, self.rng, self.degrade_version)
        mask = self.masker.skin(soft) if self.mask_on_degraded else skin[y0:y0 + c, x0:x0 + c]   # what inference sees
        return soft, tgt, mask


def synthetic_flow_grid(h: int, w: int, rng: random.Random) -> torch.Tensor:
    """Plan D (panel consensus): an analytic camera/subject motion as a grid_sample grid (1,h,w,2 in
    [-1,1]) — global similarity (rot ±2°, scale 0.98-1.03, shift ±6 px) plus a low-frequency
    deformation (4x4 control grid, 0-10 px). No learned flow estimator needed: the network must be
    invariant to small motion, not predict it."""
    ang = math.radians(rng.uniform(-2, 2)); sc = rng.uniform(0.98, 1.03)
    tx = rng.uniform(-6, 6) * 2 / w; ty = rng.uniform(-6, 6) * 2 / h
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    gx = sc * (math.cos(ang) * xs - math.sin(ang) * ys) + tx
    gy = sc * (math.sin(ang) * xs + math.cos(ang) * ys) + ty
    amp = rng.uniform(0, 10)
    ctrl = torch.tensor(np.random.default_rng(rng.randrange(1 << 30)).standard_normal((1, 2, 4, 4)), dtype=torch.float32) * amp
    field = torch.nn.functional.interpolate(ctrl, size=(h, w), mode="bicubic", align_corners=True)[0]
    gx = gx + field[0] * 2 / w; gy = gy + field[1] * 2 / h
    return torch.stack([gx, gy], -1)[None]


def warp(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Backward-warp an NHWC tensor with a grid_sample grid (bilinear, zeros outside)."""
    return torch.nn.functional.grid_sample(x.permute(0, 3, 1, 2), grid.to(x.device), mode="bilinear", padding_mode="zeros", align_corners=True).permute(0, 2, 3, 1)


def temporal_pair(pairs: "Pairs") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor, np.ndarray]:
    """One sharp crop X (frame A) and its warped copy X_w (frame B), each degraded INDEPENDENTLY;
    returns softA, tgtA, maskA, softB, tgtB, grid, valid — where grid maps B → A so that
    warp(outputA, grid) should equal outputB on valid pixels."""
    softA, tgtA, maskA = pairs.sample()
    h, w = tgtA.shape[:2]
    grid = synthetic_flow_grid(h, w, pairs.rng)
    tgtB = warp(torch.from_numpy(tgtA)[None], grid)[0].numpy()
    valid = ((grid[0, ..., 0].abs() <= 1) & (grid[0, ..., 1].abs() <= 1)).float().numpy()
    valid[:8, :] = 0; valid[-8:, :] = 0; valid[:, :8] = 0; valid[:, -8:] = 0           # border erosion
    softB = degrade(np.clip(tgtB, 0, 1), pairs.rng, pairs.degrade_version)
    maskB = pairs.masker.skin(softB) if pairs.mask_on_degraded else warp(torch.from_numpy(maskA)[None, ..., None], grid)[0, ..., 0].numpy()
    return softA, tgtA, maskA, softB, tgtB, grid, valid


def temporal_loss(outA: torch.Tensor, outB: torch.Tensor, grid: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """High-pass-only Charbonnier between warp(outA) and outB on valid pixels: invented detail must
    travel with the content instead of re-rolling per frame. RGB is left to the other losses so the
    net cannot satisfy this by fading detail."""
    a = warp(outA, grid); hp_a = a - gauss(a, 2.0); hp_b = outB - gauss(outB, 2.0)
    diff = torch.sqrt((hp_a - hp_b) ** 2 + 1e-6)
    return (diff * valid[..., None]).sum() / (valid.sum() * 3 + 1e-6)


def features_for(soft: np.ndarray, skin: np.ndarray, frame_index: int) -> np.ndarray:
    geometry = NetworkGeometry.identity(soft.shape[1], soft.shape[0])
    return make_features(soft, frame_index=frame_index, geometry=geometry, skin_mask=skin, **PROFILES["standard"])


# ----------------------------------------------------------------------------- model

ACT_CEIL = 256.0   # E4M3 saturates at 448 and the gate multiplies in fp16; run7 (2026-09-06) walked activations to 5000+
ACT = {"pen": 0.0, "max": 0.0}


def act_penalty() -> tuple[torch.Tensor | float, float]:
    """Sum over every E4M3 site of mean((|v| - ceil)+ / ceil)^2 since the last reset, and the max |activation|."""
    pen, amax = ACT["pen"], ACT["max"]; ACT["pen"] = 0.0; ACT["max"] = 0.0
    amax = float(amax.item()) if torch.is_tensor(amax) else amax
    return pen, amax

def trainable_pipeline(weights: Path, device: str) -> tuple[NeuralRenderingPipeline, list[torch.Tensor], dict[str, torch.Tensor]]:
    orig = M.e4m3_round_trip
    def ste(v):                                            # straight-through estimator + FP8 envelope barrier
        if torch.is_grad_enabled() and v.requires_grad:
            a = v.abs()
            ACT["pen"] = ACT["pen"] + (torch.relu(a - ACT_CEIL) / ACT_CEIL).square().mean()
            m = a.detach().max(); ACT["max"] = m if isinstance(ACT["max"], float) else torch.maximum(ACT["max"], m)
        return v + (orig(v) - v).detach()
    M.e4m3_round_trip = ste
    pipe = NeuralRenderingPipeline.from_safetensors(weights, device=device, precision="reference")
    names = {attr: name for name, attr in pipe.model._weight_attributes.items()}
    params, by_name = [], {}
    for attr, buf in pipe.model.named_buffers():
        name = names[attr]
        if name.endswith(("attn_scale", "attn_bias", "attention_scalar", "blend_scale")):
            continue                                            # recovered constants, no gradient path
        buf.requires_grad_(True); params.append(buf); by_name[name] = buf
    return pipe, params, by_name


def compose(head: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Inference composition in torch: colour + 0.25 * half(head[..., :3]), clamped."""
    return (soft + head[..., :3].half().float() * 0.25).clamp(0, 1)


def highpass(x: torch.Tensor) -> torch.Tensor:
    blur = torch.nn.functional.avg_pool2d(x.permute(0, 3, 1, 2), 5, stride=1, padding=2, count_include_pad=False).permute(0, 2, 3, 1)
    return x - blur


def local_energy(hp: torch.Tensor, window: int = 7) -> torch.Tensor:
    """Local mean |high-pass| — the amount of fine texture around each pixel, alignment-free."""
    return torch.nn.functional.avg_pool2d(hp.abs().permute(0, 3, 1, 2), window, stride=1, padding=window // 2, count_include_pad=False).permute(0, 2, 3, 1)


class DinoPerceptual:
    """Perceptual distance on DINOv2-base patch features (1 - cosine per token). Scale-free, so
    it rewards texture that is STRUCTURALLY like the target instead of merely having the same
    high-pass energy (which run2 turned into halos on non-face content)."""

    def __init__(self, device: str):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.model(pixel_values=x).last_hidden_state[:, 1:]          # patch tokens

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            ft = self.features(tgt)
        fp = self.features(pred)
        return (1 - torch.nn.functional.cosine_similarity(fp, ft, dim=-1)).mean()


_gauss_cache: dict = {}


def gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur of an NHWC tensor (reflect padding)."""
    key = (sigma, x.device.type)
    if key not in _gauss_cache:
        r = int(math.ceil(3 * sigma)); t = torch.arange(-r, r + 1, dtype=torch.float32)
        k = torch.exp(-t * t / (2 * sigma * sigma)); k = (k / k.sum()).to(x.device)
        _gauss_cache[key] = k
    k = _gauss_cache[key]; r = (k.numel() - 1) // 2
    c = x.shape[-1]
    y = x.permute(0, 3, 1, 2)
    y = torch.nn.functional.pad(y, (r, r, r, r), mode="reflect")
    y = torch.nn.functional.conv2d(y, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    y = torch.nn.functional.conv2d(y, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return y.permute(0, 2, 3, 1)


def median3(x: torch.Tensor) -> torch.Tensor:
    """3x3 median of an NHWC tensor (removes residual codec speckle from the target bands)."""
    y = torch.nn.functional.pad(x.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="reflect")
    patches = y.unfold(2, 3, 1).unfold(3, 3, 1).reshape(*y.shape[:2], x.shape[1], x.shape[2], 9)
    return patches.median(-1).values.permute(0, 2, 3, 1)


LAP_TERMS = {"band", "var", "halo", "mottle"}
LAP_CONTRAST = 1.0   # weight of the per-band contrast (MAD) term; 4.0 diverged in the 2026-09-06 ablation


def laplacian_loss(pred: torch.Tensor, tgt: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Panel plan C. Three Laplacian bands (σ 1-2, 2-4, 4-8) matched by L1 AND by 7x7 local mean absolute deviation
    (the contrast of the band, i.e. how real the detail feels), finer bands weighted higher, target
    bands median-filtered; plus an anti-halo hinge (no MORE high-pass than the target inside a
    5 px ring around strong low-pass edges) and an anti-mottle hinge (no high-pass where the target
    is flat). Replaces the single-scale 7x7 energy term."""
    sig = [1.0, 2.0, 4.0]; bw = [2.0, 1.5, 1.0]
    gp = [gauss(pred, s) for s in sig] + [gauss(pred, 8.0)]
    gt = [gauss(tgt, s) for s in sig] + [gauss(tgt, 8.0)]
    band_l1 = pred.new_zeros(()); var_l1 = pred.new_zeros(())
    for i in range(3):
        bp = gp[i] - gp[i + 1]; bt = median3(gt[i] - gt[i + 1])
        band_l1 = band_l1 + bw[i] * ((bp - bt).abs() * w).mean()
        # local contrast of the band = 7x7 mean absolute deviation (bounded gradient). The variance-sqrt
        # form diverged (ablation 2026-09-06: sqrt'(v+1e-6) = 500 on the near-zero bands of flat skin).
        vp = local_energy(bp.abs(), 7); vt = local_energy(bt.abs(), 7)
        var_l1 = var_l1 + bw[i] * ((vp - vt).abs() * w).mean() * LAP_CONTRAST
    hp_p = pred - gp[1]; hp_t = tgt - gt[1]                                 # high-pass above σ=2
    luma = lambda x: x[..., :1] * 0.2126 + x[..., 1:2] * 0.7152 + x[..., 2:3] * 0.0722
    lp = luma(gt[1]); gy = lp[:, 1:, :, :] - lp[:, :-1, :, :]; gx = lp[:, :, 1:, :] - lp[:, :, :-1, :]
    grad = torch.zeros_like(lp); grad[:, 1:, :, :] += gy.abs(); grad[:, :, 1:, :] += gx.abs()
    edge = (grad > 0.08).float()
    edge = torch.nn.functional.max_pool2d(edge.permute(0, 3, 1, 2), 11, stride=1, padding=5).permute(0, 2, 3, 1)
    halo = (torch.relu(hp_p.abs() - hp_t.abs()) * edge).mean() * 2.0
    flat = (hp_t.abs() < 0.008).float()
    mottle = (torch.relu(hp_p.abs() - 0.008) * flat).mean() * 1.0
    total = pred.new_zeros(())
    for name, term in (("band", band_l1), ("var", var_l1), ("halo", halo), ("mottle", mottle)):
        if name in LAP_TERMS:
            total = total + term
    return total, {"lap": band_l1.item(), "lvar": var_l1.item(), "halo": halo.item(), "mottle": mottle.item()}


class BandDiscriminator(torch.nn.Module):
    """Plan E (panel): 70x70-ish PatchGAN with spectral norm, hinge loss, fed the HIGH-PASS band (3ch)
    plus the LOW-PASS as a condition (3ch) so it can only judge texture, never tone. Returns the
    patch logits and the intermediate features (for feature matching)."""

    def __init__(self, ch: int = 48):
        super().__init__()
        sn = torch.nn.utils.spectral_norm
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(sn(torch.nn.Conv2d(6, ch, 4, 2, 1)), torch.nn.LeakyReLU(0.2)),
            torch.nn.Sequential(sn(torch.nn.Conv2d(ch, ch * 2, 4, 2, 1)), torch.nn.LeakyReLU(0.2)),
            torch.nn.Sequential(sn(torch.nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)), torch.nn.LeakyReLU(0.2)),
            torch.nn.Sequential(sn(torch.nn.Conv2d(ch * 4, ch * 8, 4, 1, 1)), torch.nn.LeakyReLU(0.2)),
        ])
        self.head = sn(torch.nn.Conv2d(ch * 8, 1, 4, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feats = []
        for block in self.blocks:
            x = block(x); feats.append(x)
        return self.head(x), feats


def band_input(img: torch.Tensor) -> torch.Tensor:
    """(N,6,H,W): high-pass above σ=2 scaled ×8, and the low-pass condition."""
    low = gauss(img, 2.0)
    return torch.cat([(img - low) * 8.0, low], -1).permute(0, 3, 1, 2)


def quantize_like_output(tgt: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Kimi's catch: outputs live on the half()×0.25 residual lattice; put the REAL target on the same
    lattice or the discriminator just learns to detect quantization."""
    return (soft + ((tgt - soft) * 4.0).half().float() * 0.25).clamp(0, 1)


def loss_fn(pred: torch.Tensor, tgt: torch.Tensor, skin: torch.Tensor, weights: dict | None = None, dino: "DinoPerceptual | None" = None) -> tuple[torch.Tensor, dict]:
    """Colour fidelity on the low-pass, a weak pixel term on the high-pass, and a strong match of
    local high-pass ENERGY (so the network is rewarded for the right amount of texture rather
    than punished for texture that is not pixel-aligned, which a plain L1 averages into blur)."""
    weights = weights or {"low": 1.0, "hp": 0.5, "energy": 4.0, "dino": 0.0}
    w = 1.0 + 2.0 * skin[..., None]                             # skin counts triple
    hp_p, hp_t = highpass(pred), highpass(tgt)
    low = (((pred - hp_p) - (tgt - hp_t)).abs() * w).mean()
    hp = ((hp_p - hp_t).abs() * w).mean()
    energy = ((local_energy(hp_p) - local_energy(hp_t)).abs() * w).mean()
    total = weights["low"] * low + weights["hp"] * hp + weights["energy"] * energy
    parts = {"low": low.item(), "hp": hp.item(), "energy": energy.item()}
    if weights.get("lap", 0) > 0:
        lap, lparts = laplacian_loss(pred, tgt, w); total = total + weights["lap"] * lap; parts.update(lparts)
    if dino is not None and weights.get("dino", 0) > 0:
        d = dino(pred, tgt); total = total + weights["dino"] * d; parts["dino"] = d.item()
    return total, parts


def save_weights(by_name: dict[str, torch.Tensor], original: Path, out: Path) -> None:
    from safetensors.torch import load_file, save_file

    base = load_file(str(original))
    for name, tensor in by_name.items():
        base[name] = tensor.detach().cpu().to(base[name].dtype)
    save_file(base, str(out), metadata=LOGICAL_FORMAT)


# ----------------------------------------------------------------------------- commands

def sharpness(path: Path) -> float:
    """Mean |L - blur(1px)| on the centre 512 crop: FFHQ-1024 measures p50 1.44, soft upscaled
    Flickr crops sit below ~1.0, pore-level skin above ~1.6."""
    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size; c = min(512, w, h)
        im = im.crop(((w - c) // 2, (h - c) // 2, (w - c) // 2 + c, (h - c) // 2 + c))
        a = np.asarray(im).astype(np.float32); b = np.asarray(im.filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32)
    return float(np.abs(a - b).mean())


def cmd_build(args) -> int:
    """List sharp stills. ``DIR`` or ``DIR:CAP`` per source; ``--min-skin`` filters on a face-parsing
    skin fraction (0 = keep everything, for general-content sets like LSDIR)."""
    masker = SkinMasker(device=args.device) if args.min_skin > 0 else None
    keep = []
    for spec in args.dirs:
        directory, _, cap = spec.partition(":")
        cap = int(cap) if cap else None
        paths = sorted(p for p in Path(directory).expanduser().rglob("*") if p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg"))
        rng = random.Random(7); rng.shuffle(paths)
        taken = 0
        for path in paths:
            if cap is not None and taken >= cap:
                break
            try:
                with Image.open(path) as im:
                    w, h = im.size
            except Exception:
                continue
            if min(w, h) < args.min_side:
                continue
            if args.min_sharp > 0 and sharpness(path) < args.min_sharp:
                continue
            if masker is not None:
                frac = float(masker.skin(load_image(path)).mean())
                if frac < args.min_skin:
                    continue
            keep.append(str(path)); taken += 1
        print(f"{directory}: {taken} stills", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(keep) + "\n")
    print(f"{len(keep)} stills -> {args.out}")
    return 0


def cmd_calibrate(args) -> int:
    """Compare the synthetic degradation with REAL soft/sharp pairs (RealSR: LR and HR of the same
    scene from one DSLR at two focal lengths). Reports PSNR and high-pass energy ratio soft/sharp
    for both, so the degradation ranges can be tuned to match real optics."""
    root = Path(args.realsr).expanduser()
    scale = str(args.scale)
    hrs = sorted(p for p in root.rglob("*_HR.png") if p.parent.name == scale)
    rng = random.Random(3); rng.shuffle(hrs); hrs = hrs[: args.n]
    if not hrs:
        raise SystemExit(f"no */{scale}/*_HR.png under {root}")
    real_psnr, real_ratio, syn_psnr, syn_ratio = [], [], [], []
    for hr_path in hrs:
        lr_path = hr_path.with_name(hr_path.name.replace("_HR.png", f"_LR{scale}.png"))
        if not lr_path.exists():
            continue
        hr = load_image(hr_path); lr = load_image(lr_path)
        if lr.shape != hr.shape:
            lr = np.asarray(Image.fromarray((lr * 255 + 0.5).astype(np.uint8)).resize((hr.shape[1], hr.shape[0]), Image.BICUBIC)).astype(np.float32) / 255
        h, w = hr.shape[:2]; c = min(512, h, w); y0 = (h - c) // 2; x0 = (w - c) // 2
        hr_c = hr[y0:y0 + c, x0:x0 + c]; lr_c = lr[y0:y0 + c, x0:x0 + c]; syn_c = degrade(hr_c, rng, args.version)
        hp = lambda a: float(highpass(torch.from_numpy(a)[None]).abs().mean())
        ps = lambda a, b: 10 * np.log10(1 / max(float(np.mean((a - b) ** 2)), 1e-10))
        real_psnr.append(ps(lr_c, hr_c)); real_ratio.append(hp(lr_c) / max(hp(hr_c), 1e-6))
        syn_psnr.append(ps(syn_c, hr_c)); syn_ratio.append(hp(syn_c) / max(hp(hr_c), 1e-6))
    print(f"pairs {len(real_psnr)}")
    print(f"real  soft vs sharp: PSNR {np.mean(real_psnr):.2f} dB, hp ratio {np.mean(real_ratio):.2f} (p10 {np.percentile(real_ratio, 10):.2f}, p90 {np.percentile(real_ratio, 90):.2f})")
    print(f"synth soft vs sharp: PSNR {np.mean(syn_psnr):.2f} dB, hp ratio {np.mean(syn_ratio):.2f} (p10 {np.percentile(syn_ratio, 10):.2f}, p90 {np.percentile(syn_ratio, 90):.2f})")
    if args.h3_frames:
        # absolute skin high-pass of real H3 frames vs of degraded FFHQ face crops (the training INPUT distribution)
        masker = SkinMasker(device=args.device)
        def skin_hp(img: np.ndarray) -> float | None:
            m = masker.skin(img) > 0.9
            if m.mean() < 0.02:
                return None
            e = highpass(torch.from_numpy(img)[None])[0].abs().mean(-1).numpy()
            return float(e[m].mean()) * 255
        h3 = [v for v in (skin_hp(load_image(p)) for p in sorted(Path(args.h3_frames).expanduser().glob("*.png"))) if v is not None]
        faces = [p for p in (Path(x) for x in Path(args.dataset).read_text().split()) if "ffhq" in str(p)]
        rng2 = random.Random(5); rng2.shuffle(faces)
        train_in = [v for v in (skin_hp(degrade(load_image(p), rng2, args.version)) for p in faces[:40]) if v is not None]
        train_tg = [v for v in (skin_hp(load_image(p)) for p in faces[:40]) if v is not None]
        print(f"skin hp (×255): real H3 frames {np.mean(h3):.2f} (p10 {np.percentile(h3, 10):.2f}, p90 {np.percentile(h3, 90):.2f}, n={len(h3)}) | "
              f"degraded train inputs {np.mean(train_in):.2f} (p10 {np.percentile(train_in, 10):.2f}, p90 {np.percentile(train_in, 90):.2f}) | sharp targets {np.mean(train_tg):.2f}")
    return 0


def split(dataset: Path, holdout: int) -> tuple[list[Path], list[Path]]:
    paths = [Path(p) for p in dataset.read_text().split() if p]
    rng = random.Random(123); rng.shuffle(paths)
    return paths[holdout:], paths[:holdout]


def evaluate(pipe: NeuralRenderingPipeline, paths: list[Path], masker: SkinMasker, crop: int, n: int, device: str, degrade_version: int = 2, mask_on_degraded: bool = True) -> dict:
    pairs = Pairs(paths, crop, masker, seed=999, degrade_version=degrade_version, mask_on_degraded=mask_on_degraded)
    psnr_in, psnr_out, hp_in, hp_out, hp_tgt = [], [], [], [], []
    with torch.no_grad():
        for i in range(n):
            soft, tgt, skin = pairs.sample()
            feats = torch.from_numpy(features_for(soft, skin, i)[None]).to(device)
            head = pipe.model(feats).float()
            pred = compose(head, torch.from_numpy(soft)[None].to(device))[0].cpu().numpy()
            mse = lambda a, b: float(np.mean((a - b) ** 2))
            psnr_in.append(10 * np.log10(1 / max(mse(soft, tgt), 1e-10))); psnr_out.append(10 * np.log10(1 / max(mse(pred, tgt), 1e-10)))
            hpn = lambda a: float(highpass(torch.from_numpy(a)[None]).abs().mean()) * 255
            hp_in.append(hpn(soft)); hp_out.append(hpn(pred)); hp_tgt.append(hpn(tgt))
    return {"psnr_soft": float(np.mean(psnr_in)), "psnr_out": float(np.mean(psnr_out)),
            "hp_soft": float(np.mean(hp_in)), "hp_out": float(np.mean(hp_out)), "hp_target": float(np.mean(hp_tgt))}


def cmd_train(args) -> int:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_paths, hold_paths = split(Path(args.dataset), args.holdout)
    pipe, params, by_name = trainable_pipeline(Path(args.weights), args.device)
    masker = SkinMasker(device=args.device)
    pairs = Pairs(train_paths, args.crop, masker, seed=args.seed, degrade_version=args.degrade, mask_on_degraded=not args.mask_on_sharp)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    weights = {"low": args.w_low, "hp": args.w_hp, "energy": args.w_energy, "dino": args.w_dino, "lap": args.w_lap}
    global LAP_TERMS, LAP_CONTRAST; LAP_TERMS = set(t for t in args.lap_terms.split(",") if t); LAP_CONTRAST = args.lap_contrast
    dino = DinoPerceptual(args.device) if args.w_dino > 0 else None
    disc = None
    if args.w_adv > 0:
        disc = BandDiscriminator().to(args.device)
        d_opt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    def adversarial(pred: torch.Tensor, tgt: torch.Tensor, soft: torch.Tensor, step: int) -> tuple[torch.Tensor, dict]:
        """One D update (hinge) on quantized reals vs detached fakes, then the G terms: hinge + feature matching.
        Ramp: 0 for the first ``adv_warmup`` steps, linear to full over the next ``adv_ramp``."""
        real = band_input(quantize_like_output(tgt, soft)); fake = band_input(pred)
        d_real, f_real = disc(real); d_fake, _ = disc(fake.detach())
        d_loss = torch.relu(1 - d_real).mean() + torch.relu(1 + d_fake).mean()
        d_opt.zero_grad(set_to_none=True); d_loss.backward(); d_opt.step()
        ramp = 0.0 if step <= args.adv_warmup else min(1.0, (step - args.adv_warmup) / max(1, args.adv_ramp))
        g_logit, f_fake = disc(fake)
        g_adv = -g_logit.mean()
        fm = sum(torch.nn.functional.l1_loss(a, b.detach()) for a, b in zip(f_fake[:3], f_real[:3])) / 3
        return ramp * (args.w_adv * g_adv + args.w_fm * fm), {"d": d_loss.item(), "adv": g_adv.item(), "fm": fm.item(), "ramp": ramp}
    log = open(out / "train.log", "a")
    def note(msg: str) -> None:
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
    note(f"loss weights {weights}")
    note(f"degrade v{args.degrade}, mask on {'sharp' if args.mask_on_sharp else 'degraded'}")
    note(f"train {len(train_paths)} stills, holdout {len(hold_paths)}, {len(params)} trainable tensors, "
         f"{sum(p.numel() for p in params) / 1e6:.1f}M params, crop {args.crop}, batch {args.batch}, lr {args.lr}")
    base_eval = evaluate(pipe, hold_paths, masker, args.crop, args.eval_n, args.device, args.degrade, not args.mask_on_sharp)
    note(f"step 0 eval {json.dumps({k: round(v, 3) for k, v in base_eval.items()})}")
    started = time.time()
    import math
    for step in range(1, args.steps + 1):
        if args.w_temporal > 0:
            pairsB = [temporal_pair(pairs) for _ in range(args.batch)]
            softA = np.stack([q[0] for q in pairsB]); tgtA = np.stack([q[1] for q in pairsB]); maskA = np.stack([q[2] for q in pairsB])
            softB = np.stack([q[3] for q in pairsB]); tgtB = np.stack([q[4] for q in pairsB]); grid = torch.cat([q[5] for q in pairsB]); valid = np.stack([q[6] for q in pairsB])
            feats = torch.from_numpy(np.stack([features_for(s_, k_, step * args.batch + i) for i, (s_, k_) in enumerate(list(zip(softA, maskA)) + list(zip(softB, [pairs.masker.skin(sb) if pairs.mask_on_degraded else m for sb, m in zip(softB, maskA)])))])).to(args.device)
            soft = torch.from_numpy(np.concatenate([softA, softB])).to(args.device)
            tgt = torch.from_numpy(np.concatenate([tgtA, tgtB])).to(args.device)
            skin = torch.from_numpy(np.concatenate([maskA, maskA])).to(args.device)
            head = pipe.model(feats).float()
            pred = compose(head, soft)
            loss, parts = loss_fn(pred, tgt, skin, weights, dino)
            act, amax = act_penalty(); loss = loss + args.w_act * act; parts["act"] = float(act.detach()) if torch.is_tensor(act) else act; parts["amax"] = amax
            n = args.batch
            t_loss = temporal_loss(pred[:n], pred[n:], grid.to(args.device), torch.from_numpy(valid).to(args.device))
            loss = loss + args.w_temporal * t_loss; parts["temporal"] = t_loss.item()
            if disc is not None:
                g_loss, gparts = adversarial(pred, tgt, soft, step); loss = loss + g_loss; parts.update(gparts)
            optimizer.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward(); gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if torch.isfinite(gnorm):
                    optimizer.step()
                else:
                    note(f"step {step}: non-finite grad norm, step skipped")
            else:
                note(f"step {step}: non-finite loss, step skipped")
            if args.cosine:
                frac = min(1.0, step / 100) if step <= 100 else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - 100) / max(1, args.steps - 100)))
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * frac
            if step % args.log_every == 0 or step == 1:
                note(f"step {step} loss {loss.item():.4f} " + " ".join(f"{k} {v:.4f}" for k, v in parts.items()) + f" {(time.time() - started) / step:.2f}s/step")
            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate(pipe, hold_paths, masker, args.crop, args.eval_n, args.device, args.degrade, not args.mask_on_sharp)
                note(f"step {step} eval {json.dumps({k: round(v, 3) for k, v in ev.items()})}")
                save_weights(by_name, Path(args.weights), out / f"dlssnr-ft-step{step}.safetensors")
                save_weights(by_name, Path(args.weights), out / "dlssnr-ft-latest.safetensors")
            continue
        if args.cosine:                                    # 100-step warmup, cosine to 5% of the base lr
            frac = min(1.0, step / 100) if step <= 100 else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - 100) / max(1, args.steps - 100)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * frac
        batch = [pairs.sample() for _ in range(args.batch)]
        feats = torch.from_numpy(np.stack([features_for(s, k, step * args.batch + i) for i, (s, _, k) in enumerate(batch)])).to(args.device)
        soft = torch.from_numpy(np.stack([b[0] for b in batch])).to(args.device)
        tgt = torch.from_numpy(np.stack([b[1] for b in batch])).to(args.device)
        skin = torch.from_numpy(np.stack([b[2] for b in batch])).to(args.device)
        head = pipe.model(feats).float()
        pred = compose(head, soft)
        loss, parts = loss_fn(pred, tgt, skin, weights, dino)
        act, amax = act_penalty(); loss = loss + args.w_act * act; parts["act"] = float(act.detach()) if torch.is_tensor(act) else act; parts["amax"] = amax
        if disc is not None:
            g_loss, gparts = adversarial(pred, tgt, soft, step); loss = loss + g_loss; parts.update(gparts)
        optimizer.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if torch.isfinite(gnorm):
                optimizer.step()
            else:
                note(f"step {step}: non-finite grad norm, step skipped")
        else:
            note(f"step {step}: non-finite loss, step skipped")
        if step % args.log_every == 0 or step == 1:
            note(f"step {step} loss {loss.item():.4f} " + " ".join(f"{k} {v:.4f}" for k, v in parts.items()) + f" {(time.time() - started) / step:.2f}s/step")
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(pipe, hold_paths, masker, args.crop, args.eval_n, args.device, args.degrade, not args.mask_on_sharp)
            note(f"step {step} eval {json.dumps({k: round(v, 3) for k, v in ev.items()})}")
            save_weights(by_name, Path(args.weights), out / f"dlssnr-ft-step{step}.safetensors")
            save_weights(by_name, Path(args.weights), out / "dlssnr-ft-latest.safetensors")
    return 0


def cmd_eval(args) -> int:
    _, hold_paths = split(Path(args.dataset), args.holdout)
    masker = SkinMasker(device=args.device)
    for weights in [args.weights] + ([args.weights2] if args.weights2 else []):
        pipe = NeuralRenderingPipeline.from_safetensors(weights, device=args.device, precision="reference")
        ev = evaluate(pipe, hold_paths, masker, args.crop, args.n, args.device)
        print(f"{Path(weights).name}: {json.dumps({k: round(v, 3) for k, v in ev.items()})}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build"); b.add_argument("dirs", nargs="+"); b.add_argument("--out", required=True)
    b.add_argument("--min-skin", type=float, default=0.05); b.add_argument("--min-side", type=int, default=768); b.add_argument("--device", default="mps")
    b.add_argument("--min-sharp", type=float, default=0.0, help="drop stills whose centre-crop sharpness is below this (1.6 keeps the sharp ~40%% of FFHQ-1024)")
    c = sub.add_parser("calibrate"); c.add_argument("--realsr", required=True); c.add_argument("--n", type=int, default=60)
    c.add_argument("--scale", type=int, default=2, help="RealSR zoom factor folder to compare against (2 = mild optical softness, 4 = strong)")
    c.add_argument("--version", type=int, default=2, help="degradation version (1 = JPEG chain, 2 = x264 mixture)")
    c.add_argument("--h3-frames", default=None, help="folder of real H3 frame PNGs: compare their skin high-pass with the degraded training inputs")
    c.add_argument("--dataset", default="ab/ft/dataset2.txt"); c.add_argument("--device", default="mps")
    t = sub.add_parser("train"); t.add_argument("--dataset", required=True); t.add_argument("--weights", required=True); t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=1500); t.add_argument("--batch", type=int, default=2); t.add_argument("--crop", type=int, default=256)
    t.add_argument("--lr", type=float, default=2e-5); t.add_argument("--seed", type=int, default=0); t.add_argument("--holdout", type=int, default=12)
    t.add_argument("--eval-every", type=int, default=250); t.add_argument("--eval-n", type=int, default=24); t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--device", default="mps")
    t.add_argument("--cosine", action="store_true", help="100-step warmup then cosine decay of the learning rate to 5%%")
    t.add_argument("--degrade", type=int, default=2, help="degradation version: 1 = JPEG chain (runs 1-5), 2 = x264 mixture (panel)")
    t.add_argument("--mask-on-sharp", action="store_true", help="compute the skin mask on the sharp target (runs 1-5) instead of the degraded input")
    t.add_argument("--w-low", type=float, default=1.0); t.add_argument("--w-hp", type=float, default=0.5)
    t.add_argument("--w-energy", type=float, default=4.0); t.add_argument("--w-dino", type=float, default=0.0, help="DINOv2 perceptual weight (0 = off)")
    t.add_argument("--lap-contrast", type=float, default=1.0, help="multiplier on the per-band contrast term inside --w-lap")
    t.add_argument("--w-lap", type=float, default=0.0, help="multi-scale Laplacian + variance + anti-halo/anti-mottle hinges (plan C); use with --w-energy 0")
    t.add_argument("--lap-terms", default="band,var,halo,mottle", help="which plan-C sub-terms are active (ablation)")
    t.add_argument("--w-act", type=float, default=10.0, help="FP8 envelope barrier: penalty on |activation| above 256 at every E4M3 site (0 = off)")
    t.add_argument("--w-temporal", type=float, default=0.0, help="plan D: synthetic-flow two-frame consistency on the high-pass (0 = off); doubles the forward cost")
    t.add_argument("--w-adv", type=float, default=0.0, help="plan E: band-limited PatchGAN hinge weight on the generator (0 = off; start 0.005)")
    t.add_argument("--w-fm", type=float, default=1.0, help="plan E: discriminator feature-matching weight")
    t.add_argument("--adv-warmup", type=int, default=200); t.add_argument("--adv-ramp", type=int, default=500)
    e = sub.add_parser("eval"); e.add_argument("--dataset", required=True); e.add_argument("--weights", required=True); e.add_argument("--weights2", default=None)
    e.add_argument("--n", type=int, default=24); e.add_argument("--crop", type=int, default=256); e.add_argument("--holdout", type=int, default=12); e.add_argument("--device", default="mps")
    args = p.parse_args()
    return {"build": cmd_build, "train": cmd_train, "eval": cmd_eval, "calibrate": cmd_calibrate}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
