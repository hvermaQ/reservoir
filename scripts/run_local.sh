#!/bin/bash
# Local end-to-end: prepare caches, then run, then aggregate.
#   ./scripts/run_local.sh configs/smoke.json 8
set -euo pipefail
CONFIG="${1:?usage: ./scripts/run_local.sh <config.json> [jobs]}"
JOBS="${2:-$(( $(python3 -c 'import os;print(os.cpu_count() or 2)') - 1 ))}"

python3 scripts/qrc.py prepare   -c "${CONFIG}"
python3 scripts/qrc.py run       -c "${CONFIG}" --jobs "${JOBS}"
python3 scripts/qrc.py aggregate -c "${CONFIG}"
