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
import json
import os
import random
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


def degrade(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Soft-video proxy calibrated on RealSR (real DSLR x2/x3 pairs keep 0.51/0.33 of the sharp
    image's high-pass energy; the first version of this kept 0.73): downscale 0.35-0.65, resample
    back, JPEG q 25-55, blur sigma 0.5-1.3."""
    h, w = image.shape[:2]
    pil = Image.fromarray((image * 255 + 0.5).astype(np.uint8))
    f = rng.uniform(0.35, 0.65)
    pil = pil.resize((max(64, int(w * f)), max(64, int(h * f))), Image.LANCZOS).resize((w, h), rng.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]))
    buf = io.BytesIO(); pil.save(buf, "JPEG", quality=rng.randint(25, 55)); buf.seek(0)
    pil = Image.open(buf).convert("RGB").filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.3)))
    return np.asarray(pil).astype(np.float32) / 255


class Pairs:
    """Random skin-biased crops of (degraded, sharp, skin mask) from a list of sharp stills."""

    def __init__(self, paths: list[Path], crop: int, masker: SkinMasker, seed: int = 0, cache: int = 48):
        self.paths, self.crop, self.masker, self.rng = paths, crop, masker, random.Random(seed)
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
        soft = degrade(tgt, self.rng)
        return soft, tgt, skin[y0:y0 + c, x0:x0 + c]


def features_for(soft: np.ndarray, skin: np.ndarray, frame_index: int) -> np.ndarray:
    geometry = NetworkGeometry.identity(soft.shape[1], soft.shape[0])
    return make_features(soft, frame_index=frame_index, geometry=geometry, skin_mask=skin, **PROFILES["standard"])


# ----------------------------------------------------------------------------- model

def trainable_pipeline(weights: Path, device: str) -> tuple[NeuralRenderingPipeline, list[torch.Tensor], dict[str, torch.Tensor]]:
    orig = M.e4m3_round_trip
    M.e4m3_round_trip = lambda v: v + (orig(v) - v).detach()   # straight-through estimator
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


def loss_fn(pred: torch.Tensor, tgt: torch.Tensor, skin: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Colour fidelity on the low-pass, a weak pixel term on the high-pass, and a strong match of
    local high-pass ENERGY (so the network is rewarded for the right amount of texture rather
    than punished for texture that is not pixel-aligned, which a plain L1 averages into blur)."""
    w = 1.0 + 2.0 * skin[..., None]                             # skin counts triple
    hp_p, hp_t = highpass(pred), highpass(tgt)
    low = (((pred - hp_p) - (tgt - hp_t)).abs() * w).mean()
    hp = ((hp_p - hp_t).abs() * w).mean()
    energy = ((local_energy(hp_p) - local_energy(hp_t)).abs() * w).mean()
    return low + 0.5 * hp + 4.0 * energy, {"low": low.item(), "hp": hp.item(), "energy": energy.item()}


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
        hr_c = hr[y0:y0 + c, x0:x0 + c]; lr_c = lr[y0:y0 + c, x0:x0 + c]; syn_c = degrade(hr_c, rng)
        hp = lambda a: float(highpass(torch.from_numpy(a)[None]).abs().mean())
        ps = lambda a, b: 10 * np.log10(1 / max(float(np.mean((a - b) ** 2)), 1e-10))
        real_psnr.append(ps(lr_c, hr_c)); real_ratio.append(hp(lr_c) / max(hp(hr_c), 1e-6))
        syn_psnr.append(ps(syn_c, hr_c)); syn_ratio.append(hp(syn_c) / max(hp(hr_c), 1e-6))
    print(f"pairs {len(real_psnr)}")
    print(f"real  soft vs sharp: PSNR {np.mean(real_psnr):.2f} dB, hp ratio {np.mean(real_ratio):.2f} (p10 {np.percentile(real_ratio, 10):.2f}, p90 {np.percentile(real_ratio, 90):.2f})")
    print(f"synth soft vs sharp: PSNR {np.mean(syn_psnr):.2f} dB, hp ratio {np.mean(syn_ratio):.2f} (p10 {np.percentile(syn_ratio, 10):.2f}, p90 {np.percentile(syn_ratio, 90):.2f})")
    return 0


def split(dataset: Path, holdout: int) -> tuple[list[Path], list[Path]]:
    paths = [Path(p) for p in dataset.read_text().split() if p]
    rng = random.Random(123); rng.shuffle(paths)
    return paths[holdout:], paths[:holdout]


def evaluate(pipe: NeuralRenderingPipeline, paths: list[Path], masker: SkinMasker, crop: int, n: int, device: str) -> dict:
    pairs = Pairs(paths, crop, masker, seed=999)
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
    pairs = Pairs(train_paths, args.crop, masker, seed=args.seed)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    log = open(out / "train.log", "a")
    def note(msg: str) -> None:
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
    note(f"train {len(train_paths)} stills, holdout {len(hold_paths)}, {len(params)} trainable tensors, "
         f"{sum(p.numel() for p in params) / 1e6:.1f}M params, crop {args.crop}, batch {args.batch}, lr {args.lr}")
    base_eval = evaluate(pipe, hold_paths, masker, args.crop, args.eval_n, args.device)
    note(f"step 0 eval {json.dumps({k: round(v, 3) for k, v in base_eval.items()})}")
    started = time.time()
    for step in range(1, args.steps + 1):
        batch = [pairs.sample() for _ in range(args.batch)]
        feats = torch.from_numpy(np.stack([features_for(s, k, step * args.batch + i) for i, (s, _, k) in enumerate(batch)])).to(args.device)
        soft = torch.from_numpy(np.stack([b[0] for b in batch])).to(args.device)
        tgt = torch.from_numpy(np.stack([b[1] for b in batch])).to(args.device)
        skin = torch.from_numpy(np.stack([b[2] for b in batch])).to(args.device)
        head = pipe.model(feats).float()
        loss, parts = loss_fn(compose(head, soft), tgt, skin)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == 1:
            note(f"step {step} loss {loss.item():.4f} low {parts['low']:.4f} hp {parts['hp']:.4f} energy {parts['energy']:.4f} {(time.time() - started) / step:.2f}s/step")
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(pipe, hold_paths, masker, args.crop, args.eval_n, args.device)
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
    t = sub.add_parser("train"); t.add_argument("--dataset", required=True); t.add_argument("--weights", required=True); t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=1500); t.add_argument("--batch", type=int, default=2); t.add_argument("--crop", type=int, default=256)
    t.add_argument("--lr", type=float, default=2e-5); t.add_argument("--seed", type=int, default=0); t.add_argument("--holdout", type=int, default=12)
    t.add_argument("--eval-every", type=int, default=250); t.add_argument("--eval-n", type=int, default=24); t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--device", default="mps")
    e = sub.add_parser("eval"); e.add_argument("--dataset", required=True); e.add_argument("--weights", required=True); e.add_argument("--weights2", default=None)
    e.add_argument("--n", type=int, default=24); e.add_argument("--crop", type=int, default=256); e.add_argument("--holdout", type=int, default=12); e.add_argument("--device", default="mps")
    args = p.parse_args()
    return {"build": cmd_build, "train": cmd_train, "eval": cmd_eval, "calibrate": cmd_calibrate}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
