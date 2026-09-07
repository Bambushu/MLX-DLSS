#!/usr/bin/env bash
# Steps 2-5 of the best-weights plan (2026-09-07): audits -> rank -> soups -> upscaler bake. One GPU job at a time.
set -u; cd "$(dirname "$0")/.."
PY=.venv/bin/python; D=ab/ft/dataset2.txt; export MLXDLSS_TORCH_CHUNK_TOKENS=0
log(){ echo "$(date '+%H:%M:%S') $*"; }
while pgrep -f "scripts/sweep_hp.sh|finetune.py train" >/dev/null; do sleep 30; done
log "GPU free; step 2: audits"
# (a) fp16 save vs live model: eval the saved lr6e6 step-800 checkpoint with the training split; compare with the logged eval
log "audit a: saved checkpoint eval (logged: $(grep 'step 800 eval' ab/ft/sw_lr6e6/stdout.log | cut -c1-120))"
$PY scripts/finetune.py eval --dataset $D --weights ab/ft/sw_lr6e6/dlssnr-ft-step800.safetensors --n 32 --crop 256 --holdout 24 --device mps 2>&1 | grep -v Loading | tail -2
# (b) Metal package vs torch for a fine-tune (no skin mask on either side)
log "audit b: Metal package vs torch"
A="--device mps --precision fast --processing-scale 1 --detail-strength 1 --colour-strength 1"
$PY -m mlxdlss.cli run --weights weights/dlssnr-weights-logical.safetensors --input ab/nr_in.png --output ab/audit_torch_stock.png $A >/dev/null 2>&1
$PY -m mlxdlss.cli run --weights weights/dlssnr-ft-real-v2.safetensors --input ab/nr_in.png --output ab/audit_torch_v2.png $A >/dev/null 2>&1
.build/release/mlxdlss render-image ab/nr_in.png weights/NeuralRendering.dlssmodel --output ab/audit_metal_stock.png --execution metal-fused --precision float16 --processing-scale 1 --detail-strength 1 --colour-strength 1 >/dev/null 2>&1
.build/release/mlxdlss render-image ab/nr_in.png weights/NeuralRendering-ft-real-v2.dlssmodel --output ab/audit_metal_v2.png --execution metal-fused --precision float16 --processing-scale 1 --detail-strength 1 --colour-strength 1 >/dev/null 2>&1
$PY - <<'PY'
import numpy as np; from PIL import Image
L=lambda f: np.asarray(Image.open(f).convert("RGB")).astype(float)
ts,tv,ms,mv,i=[L(f"ab/audit_{k}.png") for k in ("torch_stock","torch_v2","metal_stock","metal_v2")]+[L("ab/nr_in.png")]
p=lambda a,b: round(10*np.log10(255**2/max(1e-9,((a-b)**2).mean())),2)
print(f"PSNR torch-vs-metal: stock {p(ts,ms)} dB, v2 {p(tv,mv)} dB | fine-tune effect (v2 vs stock): torch {p(tv,ts)} dB, metal {p(mv,ms)} dB | v2 vs input: torch {p(tv,i)}, metal {p(mv,i)}")
PY
# (c) frozen tensors: list them (recovered constants, no gradient path by construction)
$PY - <<'PY'
from safetensors import safe_open
with safe_open("weights/dlssnr-ft-real-v2.safetensors","pt") as f: ks=list(f.keys())
fr=[k for k in ks if k.endswith(("attn_scale","attn_bias","attention_scalar","blend_scale"))]
import collections; print("audit c: frozen", len(fr), "of", len(ks), "tensors:", dict(collections.Counter(k.rsplit('.',1)[1] for k in fr)))
PY
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
