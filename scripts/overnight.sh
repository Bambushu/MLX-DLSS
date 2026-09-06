#!/usr/bin/env bash
# Overnight queue (2026-09-06): run4 long fine-tune -> loss ablation -> Metal tail jobs. No gates.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; W=weights/dlssnr-weights-logical.safetensors; D=ab/ft/dataset2.txt
log(){ echo "$(date '+%H:%M:%S') $*"; }

log "run4: 6000 steps, crop 288, cosine, energy 2 dino 1"
mkdir -p ab/ft/run4
$PY scripts/finetune.py train --dataset $D --weights $W --out ab/ft/run4 --steps 6000 --crop 288 --lr 1e-5 --cosine \
  --holdout 24 --eval-every 1000 --eval-n 32 --w-energy 2.0 --w-dino 1.0 > ab/ft/run4/stdout.log 2>&1
log "run4 exit $?"

log "ablation run5: 2500 steps, crop 256, energy 1 dino 2"
mkdir -p ab/ft/run5
$PY scripts/finetune.py train --dataset $D --weights $W --out ab/ft/run5 --steps 2500 --crop 256 --lr 1e-5 \
  --holdout 24 --eval-every 500 --eval-n 32 --w-energy 1.0 --w-dino 2.0 > ab/ft/run5/stdout.log 2>&1
log "run5 exit $?"

log "tail: full clips through v1 (Metal temporal)"
mkdir -p ab/overnight
M=weights/NeuralRendering-ft-real-v1.dlssmodel; B=.build/release/mlxdlss
$PY -m mlxdlss.video_cli convert ab/src_faces.mp4 ab/overnight/faces15s_v1.mp4 --backend mlxdlss --model $M --mlxdlss $B --temporal --overwrite >> ab/overnight/tail.log 2>&1
$PY -m mlxdlss.video_cli convert ~/ComfyUI-h3/output/video/h3_long50s_00001_.mp4 ab/overnight/long50s_v1.mp4 --backend mlxdlss --model $M --mlxdlss $B --temporal --overwrite >> ab/overnight/tail.log 2>&1
$PY -m mlxdlss.video_cli convert ab/cut12.mp4 ab/overnight/cut12_v1.mp4 --backend mlxdlss --model $M --mlxdlss $B --temporal --overwrite >> ab/overnight/tail.log 2>&1
$PY -m mlxdlss.video_cli convert ~/renderpod/h3/upscale-in/golf-concept1.mp4 ab/overnight/golf_v1.mp4 --backend mlxdlss --model $M --mlxdlss $B --temporal --overwrite >> ab/overnight/tail.log 2>&1
log "tail: v1 dial sweep on stills"
for img in ab/nr_in.png ab/krea/ray_window_rain_in.png; do n=$(basename ${img%.png}); for sc in 1 2; do for det in 1 2; do for col in 0.5 1; do
  $PY -m mlxdlss.cli run --weights weights/dlssnr-ft-real-v1.safetensors --input $img --output ab/overnight/${n}_v1_s${sc}_d${det}_c${col}.png --device mps --precision fast --processing-scale $sc --detail-strength $det --colour-strength $col --auto-mask skin >> ab/overnight/tail.log 2>&1
done; done; done; done
log "OVERNIGHT DONE"
