#!/usr/bin/env bash
# Continue only after the tuning comparison and its publication are finished.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_LATENTSPEC_FUSED_CACHED_INIT=1
exec 9>"$ROOT/rebuttal/output/full-continuation.lock"
flock -n 9
MARKER="$ROOT/rebuttal/output/tuning-published.complete.json"
echo "WAIT_TUNING_PUBLICATION $(date -u +%FT%TZ)"
while [[ ! -f "$MARKER" ]]; do sleep 30; done
"$ROOT/.venv/bin/python" - "$MARKER" <<'PY'
import json
from pathlib import Path
import sys
record = json.loads(Path(sys.argv[1]).read_text())
assert record['phase'] == 'formal_flashinfer'
assert record['paired_cells'] == 150
assert record['wins'] > record['paired_cells'] / 2
assert record['fixed_paired_cells'] == 120
assert record['fixed_wins'] > record['fixed_paired_cells'] / 2
assert record['original_doc_verified'] is True
assert record['all_raw_results_acknowledged'] is True
PY
echo "START_FULL_PREFLIGHT $(date -u +%FT%TZ)"
"$ROOT/.venv/bin/python" rebuttal/probe_init_fusion.py --production \
  --hidden-size 4096 --output rebuttal/output/init-fusion-8b-probe.json
"$ROOT/.venv/bin/python" -u rebuttal/full_suite.py --preflight
echo "START_FULL_MEASUREMENTS $(date -u +%FT%TZ)"
exec "$ROOT/.venv/bin/python" -u rebuttal/full_suite.py
