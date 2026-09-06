#!/usr/bin/env bash
# Plan-C stability ablation: 260 steps each from v2 at lr 3e-6 (same seed => same samples); which sub-term diverges?
set -u; cd "$(dirname "$0")/.."
PY=.venv/bin/python; D=ab/ft/dataset2.txt; W=weights/dlssnr-ft-real-v2.safetensors
run(){ name=$1; shift; rm -rf ab/ft/abl_$name; mkdir -p ab/ft/abl_$name
  $PY scripts/finetune.py train --dataset $D --weights $W --out ab/ft/abl_$name --steps 260 --crop 288 --lr 3e-6 --degrade 2 --holdout 24 --eval-every 260 --eval-n 16 --log-every 25 --w-dino 1.0 "$@" > ab/ft/abl_$name/stdout.log 2>&1
  echo "$(date '+%H:%M:%S') $name: $(grep -c skipped ab/ft/abl_$name/stdout.log) skipped | losses: $(grep -E '^step (25|100|175|225|250) loss' ab/ft/abl_$name/stdout.log | awk '{printf "%s:%s ", $2, $4}') | $(grep 'step 260 eval' ab/ft/abl_$name/stdout.log | cut -c1-90)"; }
run control --w-energy 2.0 --w-lap 0
run band    --w-energy 0 --w-lap 1.0 --lap-terms band
run bandvar --w-energy 0 --w-lap 1.0 --lap-terms band,var
run hinges  --w-energy 0 --w-lap 1.0 --lap-terms band,halo,mottle
run all     --w-energy 0 --w-lap 1.0 --lap-terms band,var,halo,mottle
echo "ABLATION DONE"
