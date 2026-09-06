#!/usr/bin/env python
"""Which upscaler feeds the neural re-detail best? Ground-truth bake.

  upscale_bake.py --clips ab/clips12/*.mp4 --weights weights/X.safetensors [--scale 1.5] [--frames 3] [--out ab/upscale_bake.md]

Per frame: original -> downscale by 1/scale (Lanczos, the "soft source") -> upscale back with each
method -> neural re-detail (fine-tune recipe: scale 1, detail 1, colour 1, skin mask) -> compare
with the ORIGINAL: PSNR, LPIPS, DISTS (full reference) + MUSIQ / TOPIQ (no reference) + a halo score
(high-pass excess over the original inside a 5 px edge ring). Methods: bicubic, lanczos, and every
spandrel model in --models-dir (2x/4x nets are run at their scale, then Lanczos-resampled to target).
Also reports each method WITHOUT re-detail so the upscaler's own contribution is visible."""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from mlxdlss.pipeline import NeuralRenderingPipeline  # noqa: E402
from mlxdlss.automask import SkinMasker  # noqa: E402


def frames_of(clip: Path, n: int) -> list[np.ndarray]:
    w, h, nb = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames", "-of", "csv=p=0", str(clip)], capture_output=True, text=True).stdout.strip().split(",")[:3]
    w, h, nb = int(w), int(h), int(nb)
    idx = [int(round(k * (nb - 1) / max(1, n - 1))) for k in range(n)] if n > 1 else [nb // 2]
    out = []
    for i in idx:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf", f"select=eq(n\\,{i})", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        out.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
    return out


def resample(img: np.ndarray, size: tuple[int, int], method) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize(size, method))


def run_model(model, img: np.ndarray, device: str) -> np.ndarray:
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(device)
    with torch.no_grad():
        y = model(t).clamp(0, 1)
    return (y[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def halo(out: np.ndarray, ref: np.ndarray) -> float:
    """Mean positive high-pass excess of OUT over REF inside a 5 px ring around REF's strong edges."""
    def hp(x):
        g = Image.fromarray(x).convert("L"); return np.asarray(g).astype(np.float32) - np.asarray(g.filter(ImageFilter.GaussianBlur(2))).astype(np.float32)
    ho, hr = np.abs(hp(out)), np.abs(hp(ref))
    edge = np.asarray(Image.fromarray(ref).convert("L").filter(ImageFilter.FIND_EDGES)).astype(np.float32) > 40
    ring = np.asarray(Image.fromarray((edge * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(11))) > 0
    return float(np.clip(ho - hr, 0, None)[ring].mean()) if ring.any() else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clips", nargs="+", required=True); p.add_argument("--weights", required=True)
    p.add_argument("--scale", type=float, default=1.5); p.add_argument("--frames", type=int, default=3)
    p.add_argument("--models-dir", default=str(Path.home() / "ComfyUI-h3/models/upscale_models")); p.add_argument("--device", default="mps")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    import pyiqa, spandrel
    fr = {n: pyiqa.create_metric(n, device=a.device) for n in ("lpips", "dists")}
    nr = {n: pyiqa.create_metric(n, device=a.device) for n in ("musiq", "topiq_nr")}
    models = {Path(f).stem: spandrel.ModelLoader().load_from_file(f).eval().to(a.device) for f in sorted(glob.glob(a.models_dir + "/*.safetensors") + glob.glob(a.models_dir + "/*.pth"))}
    pipe = NeuralRenderingPipeline.from_safetensors(a.weights, device=a.device, precision="fast"); masker = SkinMasker(device=a.device)
    to_t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().div(255).to(a.device)
    scores: dict[str, dict[str, list[float]]] = {}
    def record(name, out, ref):
        s = scores.setdefault(name, {})
        with torch.no_grad():
            vals = {"psnr": 10 * np.log10(255 ** 2 / max(1e-6, ((out.astype(np.float32) - ref.astype(np.float32)) ** 2).mean())),
                    **{k: float(m(to_t(out), to_t(ref))) for k, m in fr.items()}, **{k: float(m(to_t(out))) for k, m in nr.items()}, "halo": halo(out, ref)}
        for k, v in vals.items(): s.setdefault(k, []).append(v)
    for clip in a.clips:
        for ref in frames_of(Path(clip), a.frames):
            H, W = ref.shape[:2]; small = resample(ref, (round(W / a.scale), round(H / a.scale)), Image.LANCZOS)
            ups = {"bicubic": resample(small, (W, H), Image.BICUBIC), "lanczos": resample(small, (W, H), Image.LANCZOS)}
            for n, m in models.items():
                big = run_model(m, small, a.device); ups[n] = resample(big, (W, H), Image.LANCZOS) if big.shape[:2] != (H, W) else big
            for n, up in ups.items():
                record(n, up, ref)
                img = up.astype(np.float32) / 255
                out = pipe.enhance(img, processing_scale=1, detail_strength=1.0, colour_strength=1.0, skin_mask=masker.mask(img, floor=0.0)).image
                record(n + "+nr", (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), ref)
        print(Path(clip).stem, "done", flush=True)
    keys = ["psnr", "lpips", "dists", "musiq", "topiq_nr", "halo"]
    table = "| method | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys) + "\n" + "\n".join(f"| {n} | " + " | ".join(f"{np.mean(s[k]):.3f}" for k in keys) + " |" for n, s in scores.items())
    print("\n" + table)
    if a.out:
        with a.out.open("a") as f: f.write(f"\n### upscale bake x{a.scale}, {len(a.clips)} clips x {a.frames} frames, re-detail = {Path(a.weights).stem}\n\n{table}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
