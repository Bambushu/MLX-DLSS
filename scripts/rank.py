#!/usr/bin/env python
"""Rank renderer weight sets by objective quality on real content.

  rank.py --weights A.safetensors B.safetensors ... --stills DIR_OR_LIST [--n 24] [--realsr ROOT] [--device mps] [--out table.md]

No-reference (pyiqa): MUSIQ, CLIP-IQA, TOPIQ-NR (higher = better); computed on the RENDERED still.
Full-reference on RealSR pairs: LPIPS and DISTS of render(LR) against HR (lower = better).
Recipe = the fine-tune recipe (scale 1, detail 1, colour 1, skin auto-mask); pass --stock for the stock
recipe (scale 2, colour 0.5). Every metric is also reported for the INPUT so the delta is visible."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from mlxdlss.pipeline import NeuralRenderingPipeline  # noqa: E402
from mlxdlss.automask import SkinMasker  # noqa: E402


def load(path: Path, max_side: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        k = max_side / max(im.size); im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def to_t(img: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1)[None].to(device)


def list_stills(spec: Path, n: int) -> list[Path]:
    if spec.is_file() and spec.suffix == ".txt":
        paths = [Path(l.strip()) for l in spec.read_text().splitlines() if l.strip()]
    else:
        paths = sorted(p for p in spec.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    return paths[:n]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", type=Path, required=True); p.add_argument("--stills", type=Path, required=True)
    p.add_argument("--n", type=int, default=24); p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--realsr", type=Path, default=None, help="RealSR V3 root: pairs Canon/Nikon */LR2/*.png vs HR")
    p.add_argument("--device", default="mps"); p.add_argument("--stock", action="store_true"); p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    import pyiqa
    nr = {name: pyiqa.create_metric(name, device=a.device) for name in ("musiq", "clipiqa", "topiq_nr")}
    fr = {name: pyiqa.create_metric(name, device=a.device) for name in ("lpips", "dists")} if a.realsr else {}
    recipe = dict(processing_scale=2, detail_strength=1.0, colour_strength=0.5) if a.stock else dict(processing_scale=1, detail_strength=1.0, colour_strength=1.0)
    masker = None if a.stock else SkinMasker(device=a.device)
    stills = list_stills(a.stills, a.n)
    pairs = []
    if a.realsr:
        for lr in sorted(a.realsr.rglob("*_LR2.png"))[: a.n]:
            hr = lr.with_name(lr.name.replace("_LR2", "_HR"))
            if hr.exists(): pairs.append((lr, hr))
    rows = []
    def score(name, render):
        vals = {k: [] for k in list(nr) + list(fr)}
        for s in stills:
            img = load(s, a.max_side); out = render(img) if render else img
            t = to_t(out, a.device)
            with torch.no_grad():
                for k, m in nr.items(): vals[k].append(float(m(t)))
        for lr, hr in pairs:
            img = load(lr, a.max_side); out = render(img) if render else img
            hr_img = Image.open(hr).convert("RGB").resize((out.shape[1], out.shape[0]), Image.LANCZOS)
            with torch.no_grad():
                for k, m in fr.items(): vals[k].append(float(m(to_t(out, a.device), to_t(np.asarray(hr_img).astype(np.float32) / 255.0, a.device))))
        rows.append((name, {k: float(np.mean(v)) for k, v in vals.items() if v}))
        print(name, json.dumps({k: round(v, 4) for k, v in rows[-1][1].items()}), flush=True)
    score("input", None)
    for w in a.weights:
        pipe = NeuralRenderingPipeline.from_safetensors(str(w), device=a.device, precision="fast")
        score(w.stem, lambda img, pipe=pipe: np.clip(pipe.enhance(img, skin_mask=masker.mask(img, floor=0.0) if masker else None, **recipe).image, 0, 1))
        del pipe
    keys = list(rows[0][1])
    table = "| weights | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys) + "\n" + "\n".join(f"| {n} | " + " | ".join(f"{r[k]:.4f}" for k in keys) + " |" for n, r in rows)
    print("\n" + table)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("a") as f: f.write(f"\n### rank.py — {len(stills)} stills, {len(pairs)} RealSR pairs, {'stock' if a.stock else 'fine-tune'} recipe\n\n{table}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
