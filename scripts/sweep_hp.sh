#!/usr/bin/env bash
# Hyper-parameter sweep (Mike 2026-09-06: "the best product"): 800 steps each from v2, plan-C loss + FP8 barrier, EMA on.
#   lr1e-6 | lr1e-5 | late blocks only (35-70) at 3e-6 | all at 3e-6 with EMA (control)
set -u; cd "$(dirname "$0")/.."
PY=.venv/bin/python; D=ab/ft/dataset2.txt; W=weights/dlssnr-ft-real-v2.safetensors
COMMON="--dataset $D --weights $W --steps 800 --crop 256 --cosine --degrade 2 --holdout 24 --eval-every 400 --eval-n 32 --log-every 50 --w-energy 2.0 --w-lap 1.0 --lap-terms band,halo,mottle --w-dino 1.0 --w-act 10 --ema 0.999"
run(){ name=$1; shift; rm -rf ab/ft/sw_$name; mkdir -p ab/ft/sw_$name
  $PY scripts/finetune.py train $COMMON --out ab/ft/sw_$name "$@" > ab/ft/sw_$name/stdout.log 2>&1
  echo "$(date '+%H:%M:%S') $name exit $? | $(grep -c skipped ab/ft/sw_$name/stdout.log) skipped | $(grep 'step 800 eval' ab/ft/sw_$name/stdout.log | cut -c1-100)"; }
run control --lr 3e-6
run lr1e6   --lr 1e-6
run lr1e5   --lr 1e-5
run late    --lr 3e-6 --train-blocks 'block([3-6][0-9]|70)\.'
echo "SWEEP DONE"
