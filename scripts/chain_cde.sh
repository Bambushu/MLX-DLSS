#!/usr/bin/env bash
# Panel plan chain: C (run7, running) -> D (run8: temporal) -> E (run9: GAN polish). Each from the previous latest.
set -u; cd "$(dirname "$0")/.."
PY=.venv/bin/python; D=ab/ft/dataset2.txt
log(){ echo "$(date '+%H:%M:%S') $*"; }
while pgrep -f "finetune.py train --dataset ab/ft/dataset2.txt --weights weights/dlssnr-ft-real-v2.safetensors --out ab/ft/run7" >/dev/null; do sleep 20; done
log "C done; D: run8 temporal from run7"
mkdir -p ab/ft/run8
$PY scripts/finetune.py train --dataset $D --weights ab/ft/run7/dlssnr-ft-latest.safetensors --out ab/ft/run8 --steps 2500 --crop 256 --lr 3e-6 --cosine --degrade 2 \
  --holdout 24 --eval-every 500 --eval-n 32 --w-energy 2.0 --w-lap 1.0 --lap-terms band,halo,mottle --w-dino 1.0 --w-temporal 1.0 > ab/ft/run8/stdout.log 2>&1
log "D exit $?; E: run9 GAN polish from run8"
mkdir -p ab/ft/run9
$PY scripts/finetune.py train --dataset $D --weights ab/ft/run8/dlssnr-ft-latest.safetensors --out ab/ft/run9 --steps 2000 --crop 256 --lr 3e-6 --cosine --degrade 2 \
  --holdout 24 --eval-every 500 --eval-n 32 --w-energy 2.0 --w-lap 1.0 --lap-terms band,halo,mottle --w-dino 1.0 --w-temporal 0.5 --w-adv 0.005 --w-fm 1.0 --adv-warmup 200 --adv-ramp 500 > ab/ft/run9/stdout.log 2>&1
log "E exit $?"
log "CHAIN DONE"
