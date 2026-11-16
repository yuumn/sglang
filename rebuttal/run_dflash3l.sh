#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE:-static}
export SGLANG_LATENTSPEC_FUSED_CACHED_INIT=${SGLANG_LATENTSPEC_FUSED_CACHED_INIT:-1}
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-1073741824}
"$ROOT/.venv/bin/python" - <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'rebuttal')
from full_common import atomic_json
from priority_plan import make_dflash3l_plan
out=Path('rebuttal/output/dflash3l_v1')
out.mkdir(parents=True,exist_ok=True)
plan=make_dflash3l_plan()
path=out/'selection-plan.json'
if path.exists() and json.loads(path.read_text())!=plan:
    raise RuntimeError('Existing DFlash3L selection plan differs; preserve its output before changing it')
atomic_json(path,plan)
path=out/'inherited-results.json'
if not path.exists():
    atomic_json(path,{'rows':[],'models':{},'reason':'Independent three-layer checkpoint; no prior model results inherited'})
path=out/'execution-control.json'
if not path.exists():
    atomic_json(path,{'stop_after_stage':1,'reason':'Only explicitly requested DFlash3L after completed DFlash4L'})
PY
exec "$ROOT/.venv/bin/python" -u "$ROOT/rebuttal/priority_suite.py" \
  --output "$ROOT/rebuttal/output/dflash3l_v1" \
  --plan "$ROOT/rebuttal/output/dflash3l_v1/selection-plan.json"
