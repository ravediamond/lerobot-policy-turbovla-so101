#!/usr/bin/env bash

# Copyright 2026 The TurboVLA-LeRobot contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Matched-protocol training for turbovla_so101 / act / smolvla: same dataset, same steps, same
# 224px resize overlay (see resize224.yaml), logged so eval_curve.py and plot_curves.py can
# score/plot them afterward without re-running anything.
#
# Default is sequential — one policy at a time, each getting the full GPU/memory budget. Pass
# PARALLEL=1 to background all three at once instead; that's only safe with enough VRAM/unified
# memory to hold all three training states simultaneously (fine on a discrete-GPU box with headroom
# to spare, likely to OOM on a shared-memory device like a Jetson).
#
# Usage:
#   DATASET=max-chr/libero_plus_object_language_all STEPS=8000 BATCH_SIZE=64 \
#       benchmarks/run_comparison.sh
#   PARALLEL=1 benchmarks/run_comparison.sh   # background all three (needs the VRAM for it)
#
# Each run writes outputs/cmp_<policy>/ (checkpoints) and outputs/cmp_<policy>.log (stdout, parsed
# by plot_curves.py for the loss curve).

set -euo pipefail

DATASET="${DATASET:-max-chr/libero_plus_object_language_all}"
STEPS="${STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PARALLEL="${PARALLEL:-0}"
POLICIES=(turbovla_so101 act smolvla)

mkdir -p outputs

run_one() {
    local policy="$1"
    echo "launching ${policy}: ${STEPS} steps, batch ${BATCH_SIZE}, dataset ${DATASET}"
    python3 benchmarks/train_with_yaml.py benchmarks/resize224.yaml \
        --dataset.repo_id="${DATASET}" \
        --policy.type="${policy}" \
        --batch_size="${BATCH_SIZE}" \
        --steps="${STEPS}" \
        --eval_steps=0 \
        --wandb.enable=false \
        --output_dir="outputs/cmp_${policy}" \
        > "outputs/cmp_${policy}.log" 2>&1
}

if [[ "${PARALLEL}" == "1" ]]; then
    pids=()
    for policy in "${POLICIES[@]}"; do
        run_one "${policy}" &
        pids+=("$!")
    done
    echo "waiting on ${#pids[@]} runs (pids: ${pids[*]})"
    wait "${pids[@]}"
else
    for policy in "${POLICIES[@]}"; do
        run_one "${policy}"
        echo "${policy} done"
    done
fi

echo "done — see outputs/cmp_*.log and outputs/cmp_*/checkpoints"
