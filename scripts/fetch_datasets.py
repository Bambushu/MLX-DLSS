#!/usr/bin/env python
"""Fetch the run2 training data: FFHQ-1024 shards, LSDIR parquet shards, RealSR V3 (eval).

  fetch_datasets.py ROOT [--ffhq-shards 6] [--lsdir-shards 10] [--realsr] [--flickr2k] [--div8k]
Images are extracted to ROOT/ffhq/*.webp, ROOT/lsdir/*.png, ROOT/flickr2k/*.png, ROOT/div8k/*.png
(DIV8K resampled to <= 3072 px on the long side, Lanczos), RealSR to ROOT/realsr/.
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def token() -> str | None:
    p = Path.home() / ".hf_token"
    return p.read_text().strip() if p.exists() else None


def fetch_ffhq(root: Path, shards: int) -> None:
    out = root / "ffhq"; out.mkdir(parents=True, exist_ok=True)
    for i in range(shards):
        name = f"{i * 1000:05d}.tar"
        if sum(1 for p in out.glob("*.webp") if i * 1000 <= int(p.stem) < (i + 1) * 1000) >= 900:
            print(f"ffhq {name}: already extracted", flush=True); continue
        path = hf_hub_download("gaunernst/ffhq-1024-wds", name, repo_type="dataset", token=token(), cache_dir=str(root / "_hub"))
        n = 0
        with tarfile.open(path) as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".webp"):
                    (out / Path(member.name).name).write_bytes(tar.extractfile(member).read()); n += 1
        print(f"ffhq {name}: {n} images", flush=True)
        Path(path).unlink(missing_ok=True)


def fetch_lsdir(root: Path, shards: int) -> None:
    import pyarrow.parquet as pq

    out = root / "lsdir"; out.mkdir(parents=True, exist_ok=True)
    for i in range(shards):
        name = f"data/train-{i:05d}-of-00195.parquet"
        marker = out / f".shard{i:05d}.done"
        if marker.exists():
            print(f"lsdir shard {i}: already extracted", flush=True); continue
        path = hf_hub_download("danjacobellis/LSDIR", name, repo_type="dataset", token=token(), cache_dir=str(root / "_hub"))
        table = pq.read_table(path, columns=["path", "image", "w", "h"])
        n = 0
        for row in table.to_pylist():
            if min(row["w"], row["h"]) < 768:
                continue
            img = row["image"]; data = img["bytes"] if isinstance(img, dict) else img
            stem = Path(row["path"]).stem
            (out / f"{stem}.png").write_bytes(data); n += 1
        marker.touch(); Path(path).unlink(missing_ok=True)
        print(f"lsdir shard {i}: {n} images >= 768px", flush=True)


def fetch_realsr(root: Path) -> None:
    out = root / "realsr"; out.mkdir(parents=True, exist_ok=True)
    if any(out.rglob("*.png")):
        print("realsr: already extracted", flush=True); return
    path = hf_hub_download("Yuuuuuu9/realsr_drealsr", "RealSR(V3).tar.gz", repo_type="dataset", token=token(), cache_dir=str(root / "_hub"))
    with tarfile.open(path) as tar:
        tar.extractall(out, filter="data")
    print(f"realsr: extracted {sum(1 for _ in out.rglob('*.png'))} pngs", flush=True)
    Path(path).unlink(missing_ok=True)


def _fetch_zip(root: Path, repo: str, filename: str, sub: str, max_side: int | None) -> None:
    """One HF zip of stills -> ROOT/sub/*.png, optionally Lanczos-resampled so the long side <= max_side."""
    import zipfile
    from PIL import Image

    out = root / sub; out.mkdir(parents=True, exist_ok=True)
    marker = out / ".done"
    if marker.exists():
        print(f"{sub}: already extracted", flush=True); return
    path = hf_hub_download(repo, filename, repo_type="dataset", token=token(), cache_dir=str(root / "_hub"))
    n = 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            target = out / (Path(info.filename).stem + ".png")
            if target.exists():
                n += 1; continue
            with z.open(info) as f:
                im = Image.open(io.BytesIO(f.read())).convert("RGB")
            if max_side and max(im.size) > max_side:
                k = max_side / max(im.size)
                im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
            im.save(target, compress_level=1); n += 1
            if n % 200 == 0:
                print(f"{sub}: {n} images", flush=True)
    marker.touch(); Path(path).unlink(missing_ok=True)
    print(f"{sub}: {n} images", flush=True)


def fetch_flickr2k(root: Path) -> None:
    _fetch_zip(root, "yangtao9009/Flickr2K", "Flickr2K.zip", "flickr2k", None)


def fetch_div8k(root: Path) -> None:
    _fetch_zip(root, "yangtao9009/DIV8K", "DIV8K.zip", "div8k", 3072)


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path)
    p.add_argument("--ffhq-shards", type=int, default=6); p.add_argument("--lsdir-shards", type=int, default=10); p.add_argument("--realsr", action="store_true")
    p.add_argument("--flickr2k", action="store_true"); p.add_argument("--div8k", action="store_true")
    a = p.parse_args()
    fetch_ffhq(a.root, a.ffhq_shards)
    fetch_lsdir(a.root, a.lsdir_shards)
    if a.realsr:
        fetch_realsr(a.root)
    if a.flickr2k:
        fetch_flickr2k(a.root)
    if a.div8k:
        fetch_div8k(a.root)
    print("FETCH DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
