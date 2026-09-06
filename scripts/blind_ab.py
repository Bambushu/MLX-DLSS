#!/usr/bin/env python
"""Blind A/B of renderer weight sets on short clips.

  blind_ab.py render OUT_DIR --clips a.mp4 b.mp4 ... --weights NAME=path.safetensors ...   # every clip x every weight set (GPU)
  blind_ab.py pair OUT_DIR [--seed 1]        # shuffled left/right side-by-sides pair_NN.mp4 + hidden key.json
  blind_ab.py score OUT_DIR "L R = L ..."    # your verdict per pair in order; tallies wins per weight set
  blind_ab.py strips OUT_DIR [--frame 0.5]   # 100% centre crops of each pair (left|right) for a vision-model grader

Render recipe = the fine-tune recipe: scale 1, detail 1, colour 1, skin auto-mask, hp-history 0.5."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path


def sh(*args: str) -> None:
    subprocess.run(list(args), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render(a) -> int:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    weights = dict(w.split("=", 1) for w in a.weights)
    for clip in a.clips:
        for name, path in weights.items():
            target = out / f"{Path(clip).stem}__{name}.mp4"
            if target.exists():
                continue
            sh(sys.executable, "-m", "mlxdlss.video_cli", "convert", clip, str(target), "--weights", path, "--device", a.device, "--precision", "fast",
               "--processing-scale", "1", "--detail-strength", "1", "--colour-strength", "1", "--auto-mask", "skin", "--temporal", "--hp-history", "0.5", "--batch", "1")
            print("rendered", target.name, flush=True)
    (out / "weights.json").write_text(json.dumps(weights, indent=1))
    return 0


def pair(a) -> int:
    out = Path(a.out); rng = random.Random(a.seed)
    weights = json.loads((out / "weights.json").read_text())
    clips = sorted({p.name.split("__")[0] for p in out.glob("*__*.mp4")})
    combos = [(c, x, y) for c in clips for x, y in itertools.combinations(sorted(weights), 2)]
    rng.shuffle(combos)
    key = []
    for i, (clip, x, y) in enumerate(combos):
        left, right = (x, y) if rng.random() < 0.5 else (y, x)
        target = out / f"pair_{i:02d}.mp4"
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(out / f"{clip}__{left}.mp4"), "-i", str(out / f"{clip}__{right}.mp4"),
           "-filter_complex", "[0:v][1:v]hstack=inputs=2", "-an", "-c:v", "libx264", "-crf", "14", "-preset", "fast", "-pix_fmt", "yuv420p", str(target))
        key.append({"pair": target.name, "clip": clip, "left": left, "right": right})
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(key)} pairs -> {out}/pair_NN.mp4 (key hidden in key.json; do not open it before scoring)")
    return 0


def score(a) -> int:
    out = Path(a.out); key = json.loads((out / "key.json").read_text())
    verdicts = a.verdicts.replace(",", " ").split()
    if len(verdicts) != len(key):
        raise SystemExit(f"{len(key)} pairs, {len(verdicts)} verdicts")
    wins, games = {}, {}
    for k, v in zip(key, verdicts):
        for n in (k["left"], k["right"]):
            games[n] = games.get(n, 0) + 1; wins.setdefault(n, 0.0)
        if v.upper() == "L": wins[k["left"]] += 1
        elif v.upper() == "R": wins[k["right"]] += 1
        else: wins[k["left"]] += 0.5; wins[k["right"]] += 0.5
    print("| weights | wins | games | win rate |\n|---|---|---|---|")
    for n in sorted(wins, key=lambda n: -wins[n] / games[n]):
        print(f"| {n} | {wins[n]:.1f} | {games[n]} | {wins[n] / games[n]:.2f} |")
    (out / "score.json").write_text(json.dumps({"verdicts": verdicts, "wins": wins, "games": games}, indent=1))
    return 0


def strips(a) -> int:
    out = Path(a.out); key = json.loads((out / "key.json").read_text())
    for k in key:
        target = out / (Path(k["pair"]).stem + ".png")
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(out / k["pair"]), "-vf", f"select=gte(n\\,{a.frame})", "-frames:v", "1", str(target))
    print(f"{len(key)} strips written (frame {a.frame})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render"); r.add_argument("out"); r.add_argument("--clips", nargs="+", required=True); r.add_argument("--weights", nargs="+", required=True); r.add_argument("--device", default="mps")
    q = sub.add_parser("pair"); q.add_argument("out"); q.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("score"); s.add_argument("out"); s.add_argument("verdicts")
    t = sub.add_parser("strips"); t.add_argument("out"); t.add_argument("--frame", type=int, default=24)
    a = p.parse_args()
    return {"render": render, "pair": pair, "score": score, "strips": strips}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
