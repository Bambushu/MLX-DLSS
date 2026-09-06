#!/usr/bin/env python
"""Weight soup: weighted average of fine-tuned renderer checkpoints (same logical format).

  soup.py OUT.safetensors A.safetensors[:w] B.safetensors[:w] ...   (weights default 1, normalised)
Non-float tensors and the format metadata are taken from the first file."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> int:
    out, specs = Path(sys.argv[1]), sys.argv[2:]
    files, weights = [], []
    for spec in specs:
        path, _, w = spec.partition(":")
        files.append(path); weights.append(float(w) if w else 1.0)
    total = sum(weights)
    with safe_open(files[0], "pt") as f:
        meta = f.metadata()
    base = load_file(files[0])
    acc = {k: v.float() * (weights[0] / total) for k, v in base.items() if v.is_floating_point()}
    for path, w in zip(files[1:], weights[1:]):
        other = load_file(path)
        for k in acc:
            acc[k] += other[k].float() * (w / total)
    for k, v in acc.items():
        base[k] = v.to(base[k].dtype)
    save_file(base, str(out), metadata=meta)
    print(f"{out}: {len(acc)} tensors averaged from {len(files)} files, weights {[round(w / total, 3) for w in weights]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
