#!/usr/bin/env python
"""Repeatable validation for the fork (roadmap item 4).

  validate.py framegen CLIP.mp4 [...] --weights weights/framegen.safetensors [--rife DIR] [--minterpolate]
      Withheld-frame protocol: drop odd frames (fps/2), regenerate, PSNR of the generated frames
      against the withheld originals. Methods: dlss (this port), rife47 / rife426 (from a
      ComfyUI-Frame-Interpolation checkout given by --rife), ffmpeg minterpolate.
  validate.py renderer CLIP.mp4 --weights weights/dlssnr-weights-logical.safetensors [--frames 24] [--start 0]
      Temporal-mode flicker ratio (consecutive-frame diff, output / source) and high-pass detail
      energy on skin vs outside (face-parsing mask), for the Krea2 recipe with and without the auto mask.
Prints a markdown table; --out appends it to a file.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def sh(*args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), check=True, capture_output=True, **kw)


def probe(path: Path) -> tuple[int, int, float]:
    out = sh("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(path)).stdout.decode().strip().split(",")
    num, den = out[2].split("/")
    return int(out[0]), int(out[1]), float(num) / float(den)


def decode(path: Path, width: int, height: int) -> np.ndarray:
    raw = sh("ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-").stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def encode(frames: np.ndarray, path: Path, fps: float) -> None:
    h, w = frames.shape[1:3]
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
                             "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(path)], stdin=subprocess.PIPE)
    proc.stdin.write(np.ascontiguousarray(frames).tobytes()); proc.stdin.close(); proc.wait()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def luma(f: np.ndarray) -> np.ndarray:
    return f[..., 0] * 0.2126 + f[..., 1] * 0.7152 + f[..., 2] * 0.0722


# ----------------------------------------------------------------------------- frame generation

def fg_dlss(frames: np.ndarray, weights: Path, device: str) -> tuple[list[np.ndarray], float]:
    from mlxdlss.framegen import FrameGenerator

    gen = FrameGenerator.from_safetensors(weights, device=device, precision="fast")
    started = time.perf_counter(); out = []
    for start in range(0, len(frames) - 1, 4):
        window = [frames[i] for i in range(start, min(start + 5, len(frames)))]
        for pair in gen.generate_pairs(window, 2):
            out.append(pair[0])
    return out, time.perf_counter() - started


def fg_rife(frames: np.ndarray, rife_dir: Path, ckpt: str, arch: str, device: str) -> tuple[list[np.ndarray], float]:
    import types
    import torch

    comfy = types.ModuleType("comfy"); mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: torch.device(device); comfy.model_management = mm
    sys.modules.setdefault("comfy", comfy); sys.modules.setdefault("comfy.model_management", mm)
    sys.path.insert(0, str(rife_dir / "vfi_models" / "rife"))
    from rife_arch import IFNet  # type: ignore

    model = IFNet(arch_ver=arch)
    state = torch.load(rife_dir / "ckpts" / "rife" / ckpt, weights_only=False, map_location="cpu")
    missing, _unexpected = model.load_state_dict(state, strict=False)   # 4.26 ships teacher/caltime training heads
    if missing:
        raise RuntimeError(f"{ckpt}: missing keys {missing[:3]}")
    model.eval().to(device)
    scale_list = [16, 8, 4, 2, 1] if arch == "4.26" else [8, 4, 2, 1]
    started = time.perf_counter(); out = []
    with torch.inference_mode():
        for i in range(len(frames) - 1):
            a = torch.from_numpy(frames[i]).permute(2, 0, 1)[None].float().div(255).to(device)
            b = torch.from_numpy(frames[i + 1]).permute(2, 0, 1)[None].float().div(255).to(device)
            ts = torch.tensor([0.5], device=device).view(1, 1, 1, 1)
            mid = model(a, b, ts, scale_list, False, False).clamp(0, 1)[0].permute(1, 2, 0).mul(255).add(0.5).byte().cpu().numpy()
            out.append(mid)
    return out, time.perf_counter() - started


def fg_minterpolate(half: Path, width: int, height: int, fps: float, tmp: Path) -> tuple[list[np.ndarray], float]:
    out = tmp / "mint.mp4"; started = time.perf_counter()
    sh("ffmpeg", "-v", "error", "-y", "-i", str(half), "-vf", f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
       "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(out))
    seconds = time.perf_counter() - started
    frames = decode(out, width, height)
    return [frames[i] for i in range(1, len(frames), 2)], seconds


def run_framegen(args: argparse.Namespace) -> str:
    rows = ["| clip | method | mean PSNR (dB) | worst | gen fps |", "|---|---|---|---|---|"]
    for clip in args.clips:
        width, height, fps = probe(clip)
        truth = decode(clip, width, height)
        if args.frames:
            truth = truth[:args.frames]
        if len(truth) % 2 == 0:
            truth = truth[:-1]
        half = truth[::2]; withheld = truth[1::2]
        methods: dict[str, tuple[list[np.ndarray], float]] = {}
        methods["dlss"] = fg_dlss(half, args.weights, args.device)
        if args.rife:
            for ckpt, arch in (("rife47.pth", "4.7"), ("rife426.pth", "4.26")):
                if (args.rife / "ckpts" / "rife" / ckpt).exists():
                    methods[ckpt[:-4]] = fg_rife(half, args.rife, ckpt, arch, args.device)
        if args.minterpolate:
            with tempfile.TemporaryDirectory() as directory:
                tmp = Path(directory); half_path = tmp / "half.mp4"
                encode(half, half_path, fps / 2)
                methods["minterpolate"] = fg_minterpolate(half_path, width, height, fps, tmp)
        for name, (generated, seconds) in methods.items():
            scores = [psnr(g, t) for g, t in zip(generated, withheld)]
            rows.append(f"| {clip.stem} | {name} | {np.mean(scores):.2f} | {min(scores):.2f} | {len(generated) / seconds:.1f} |")
    return "\n".join(rows)


# ----------------------------------------------------------------------------- renderer

def run_renderer(args: argparse.Namespace) -> str:
    from PIL import Image, ImageFilter

    from mlxdlss.automask import SkinMasker
    from mlxdlss.pipeline import NeuralRenderingPipeline
    from mlxdlss.temporal import TemporalOptions, TemporalSession

    width, height, fps = probe(args.clip)
    src = decode(args.clip, width, height)[args.start:args.start + args.frames].astype(np.float32) / 255
    pipeline = NeuralRenderingPipeline.from_safetensors(args.weights, device=args.device, precision="fast")
    masker = SkinMasker(device=pipeline.device)
    masks = [masker.mask(f) for f in src]

    def highpass(a: np.ndarray) -> np.ndarray:
        img = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
        b = np.asarray(img.filter(ImageFilter.GaussianBlur(3))).astype(np.float32) / 255
        return np.abs(a - b).mean(2)

    def stats(frames: np.ndarray) -> tuple[float, float, float]:
        flick = np.mean([np.abs(frames[i + 1] - frames[i]).mean() for i in range(len(frames) - 1)])
        hp_skin, hp_out = [], []
        for f, m in zip(frames, masks):
            e = highpass(f); hp_skin.append(e[m > 0.9].mean()); hp_out.append(e[m < 0.1].mean())
        return float(flick), float(np.mean(hp_skin)) * 255, float(np.mean(hp_out)) * 255

    rows = ["| variant | flicker ratio | hp skin | hp outside | s/frame |", "|---|---|---|---|---|"]
    base = stats(src)
    rows.append(f"| source | 1.00 | {base[1]:.2f} | {base[2]:.2f} | - |")
    for name, use_mask in (("temporal, c0.5", False), ("temporal, c0.5, auto-mask", True)):
        session = TemporalSession(pipeline, options=TemporalOptions(colour_strength=0.5, scene_cut_threshold=0.3), motion="flow")
        started = time.perf_counter()
        out = np.stack([np.clip(session.process(f, skin_mask=m if use_mask else None), 0, 1) for f, m in zip(src, masks)])
        seconds = (time.perf_counter() - started) / len(src)
        s = stats(out)
        rows.append(f"| {name} | {s[0] / base[0]:.2f} | {s[1]:.2f} | {s[2]:.2f} | {seconds:.1f} |")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fg = sub.add_parser("framegen"); fg.add_argument("clips", nargs="+", type=Path); fg.add_argument("--weights", type=Path, required=True)
    fg.add_argument("--rife", type=Path, default=None, help="ComfyUI-Frame-Interpolation checkout (uses ckpts/rife/rife47.pth and rife426.pth when present)")
    fg.add_argument("--minterpolate", action="store_true"); fg.add_argument("--frames", type=int, default=None); fg.add_argument("--device", default="mps")
    nr = sub.add_parser("renderer"); nr.add_argument("clip", type=Path); nr.add_argument("--weights", type=Path, required=True)
    nr.add_argument("--frames", type=int, default=24); nr.add_argument("--start", type=int, default=0); nr.add_argument("--device", default="mps")
    for p in (fg, nr):
        p.add_argument("--out", type=Path, default=None, help="append the table to this markdown file")
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")
    table = run_framegen(args) if args.command == "framegen" else run_renderer(args)
    print(table)
    if args.out:
        with open(args.out, "a") as f:
            f.write(f"\n### validate.py {args.command} — {time.strftime('%Y-%m-%d %H:%M')}\n{table}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
