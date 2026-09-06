#!/usr/bin/env bash
# Plan-C ablation round 2: the contrast term at x4 diverges (even as MAD). Two stable candidates?
set -u; cd "$(dirname "$0")/.."
PY=.venv/bin/python; D=ab/ft/dataset2.txt; W=weights/dlssnr-ft-real-v2.safetensors
run(){ name=$1; shift; rm -rf ab/ft/abl_$name; mkdir -p ab/ft/abl_$name
  $PY scripts/finetune.py train --dataset $D --weights $W --out ab/ft/abl_$name --steps 260 --crop 288 --lr 3e-6 --degrade 2 --holdout 24 --eval-every 260 --eval-n 16 --log-every 25 --w-dino 1.0 "$@" > ab/ft/abl_$name/stdout.log 2>&1
  echo "$(date '+%H:%M:%S') $name: $(grep -c skipped ab/ft/abl_$name/stdout.log) skipped | losses: $(grep -E '^step (25|100|175|225|250) loss' ab/ft/abl_$name/stdout.log | awk '{printf "%s:%s ", $2, $4}') | $(grep 'step 260 eval' ab/ft/abl_$name/stdout.log | cut -c1-90)"; }
run hingesE --w-energy 2.0 --w-lap 1.0 --lap-terms band,halo,mottle
run all1    --w-energy 0   --w-lap 1.0 --lap-terms band,var,halo,mottle --lap-contrast 1.0
echo "ABLATION2 DONE"
