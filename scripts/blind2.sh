#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."; PY=.venv/bin/python; export MLXDLSS_TORCH_CHUNK_TOKENS=0
echo "$(date '+%H:%M:%S') blind2 renders at detail 2"
rm -rf ab/blind2; $PY scripts/blind_ab.py render ab/blind2 --clips ab/clips12/*.mp4 --weights crisp=weights/dlssnr-ft-real-v1-crisp.safetensors v2=weights/dlssnr-ft-real-v2.safetensors final=ab/ft/final1/dlssnr-ft-ema-step6000.safetensors --device mps --detail 2 2>&1 | grep -vE "Loading|Warning|warn" | tail -2
$PY scripts/blind_ab.py pair ab/blind2 --seed 11 --crop 480 --zoom 2 --calibrate 3 && $PY scripts/blind_ab.py strips ab/blind2 --frame 36
echo "$(date '+%H:%M:%S') BLIND2 DONE"
