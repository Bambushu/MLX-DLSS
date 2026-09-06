#!/usr/bin/env bash
# Wait for chain_cde.sh to finish, then run the hyper-parameter sweep. Nothing else touches the GPU meanwhile.
set -u; cd "$(dirname "$0")/.."
while pgrep -f "scripts/chain_cde.sh" >/dev/null; do sleep 60; done
echo "$(date '+%H:%M:%S') chain finished; starting sweep"
bash scripts/sweep_hp.sh
