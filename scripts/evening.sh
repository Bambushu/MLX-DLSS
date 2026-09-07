#!/usr/bin/env bash
# After final1: rank its checkpoints vs the incumbents, then the blind A/B renders + pairs (crisp, v2, final EMA).
set -u; cd "$(dirname "$0")/.."
PY=.venv/bin/python; export MLXDLSS_TORCH_CHUNK_TOKENS=0
log(){ echo "$(date '+%H:%M:%S') $*"; }
while pgrep -f "finetune.py train" >/dev/null; do sleep 30; done
log "final1 done; ranking its checkpoints"
$PY scripts/rank.py --weights weights/dlssnr-ft-real-v1-crisp.safetensors weights/dlssnr-ft-real-v2.safetensors ab/ft/run9/dlssnr-ft-latest.safetensors ab/ft/final1/dlssnr-ft-step3000.safetensors ab/ft/final1/dlssnr-ft-ema-step4000.safetensors ab/ft/final1/dlssnr-ft-ema-step5000.safetensors ab/ft/final1/dlssnr-ft-step6000.safetensors ab/ft/final1/dlssnr-ft-ema-step6000.safetensors --stills ab/ft/h3frames --n 31 --realsr ~/mlx-dlss-data/realsr --device mps --out ab/RESULTS.md 2>&1 | grep -vE "Loading|Warning|warn"
log "blind A/B renders: crisp / v2 / final1 EMA@6000 on ab/clips12"
rm -rf ab/blind1; $PY scripts/blind_ab.py render ab/blind1 --clips ab/clips12/*.mp4 --weights crisp=weights/dlssnr-ft-real-v1-crisp.safetensors v2=weights/dlssnr-ft-real-v2.safetensors final=ab/ft/final1/dlssnr-ft-ema-step6000.safetensors --device mps 2>&1 | grep -vE "Loading|Warning|warn" | tail -3
$PY scripts/blind_ab.py pair ab/blind1 --seed 7 && $PY scripts/blind_ab.py strips ab/blind1
log "EVENING DONE"
