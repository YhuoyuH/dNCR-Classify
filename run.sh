#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/experiment.yml}"
RESULTS_DIR="${2:-results}"

python train.py --config "${CONFIG_PATH}" --results "${RESULTS_DIR}"
