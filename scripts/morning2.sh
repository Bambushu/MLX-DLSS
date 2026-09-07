#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."
PY=.venv/bin/python; export MLXDLSS_TORCH_CHUNK_TOKENS=0
log(){ echo "$(date "+%H:%M:%S") $*"; }
log "step 3: rank checkpoints (H3 frames + RealSR)"
$PY scripts/rank.py --weights weights/dlssnr-ft-real-v2.safetensors weights/dlssnr-ft-real-v1-crisp.safetensors weights/dlssnr-ft-real-v1.safetensors ab/ft/run7/dlssnr-ft-latest.safetensors ab/ft/run8/dlssnr-ft-latest.safetensors ab/ft/run9/dlssnr-ft-latest.safetensors ab/ft/sw_control/dlssnr-ft-step800.safetensors ab/ft/sw_lr6e6/dlssnr-ft-step800.safetensors ab/ft/sw_lr6e6/dlssnr-ft-ema-step800.safetensors ab/ft/sw_lr1e5/dlssnr-ft-step800.safetensors ab/ft/sw_lr1e5/dlssnr-ft-ema-step800.safetensors ab/ft/sw_late/dlssnr-ft-step800.safetensors --stills ab/ft/h3frames --n 31 --realsr ~/mlx-dlss-data/realsr --device mps --out ab/RESULTS.md 2>&1 | grep -vE "Loading|Warning|warn"
log "step 4: soups"
mkdir -p ab/ft/soups
$PY scripts/soup.py ab/ft/soups/v2_crisp.safetensors weights/dlssnr-ft-real-v2.safetensors weights/dlssnr-ft-real-v1-crisp.safetensors
$PY scripts/soup.py ab/ft/soups/v2_lr6e6.safetensors weights/dlssnr-ft-real-v2.safetensors ab/ft/sw_lr6e6/dlssnr-ft-ema-step800.safetensors
$PY scripts/soup.py ab/ft/soups/lr6e6_run9.safetensors ab/ft/sw_lr6e6/dlssnr-ft-ema-step800.safetensors ab/ft/run9/dlssnr-ft-latest.safetensors
$PY scripts/soup.py ab/ft/soups/v2_crisp_lr6e6.safetensors weights/dlssnr-ft-real-v2.safetensors weights/dlssnr-ft-real-v1-crisp.safetensors ab/ft/sw_lr6e6/dlssnr-ft-ema-step800.safetensors
$PY scripts/rank.py --weights ab/ft/soups/v2_crisp.safetensors ab/ft/soups/v2_lr6e6.safetensors ab/ft/soups/lr6e6_run9.safetensors ab/ft/soups/v2_crisp_lr6e6.safetensors --stills ab/ft/h3frames --n 31 --realsr ~/mlx-dlss-data/realsr --device mps --out ab/RESULTS.md 2>&1 | grep -vE "Loading|Warning|warn"
log "step 5: upscaler bake (re-detail = v2)"
$PY scripts/upscale_bake.py --clips ab/clips12/*.mp4 --weights weights/dlssnr-ft-real-v2.safetensors --scale 1.5 --frames 3 --device mps --out ab/RESULTS.md 2>&1 | grep -vE "Loading|Warning|warn"
log "MORNING DONE"
