#!/usr/bin/env python3
"""Read durable matrix progress without touching the running server."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "output"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--phase", default="formal")
args = p.parse_args()
summary = {"phase": args.phase, "checked_at": datetime.now(timezone.utc).isoformat(),
           "complete": (ROOT / (args.phase + ".complete.json")).exists(), "variants": {}}
pidfile = ROOT / "matrix.pid"
if pidfile.exists():
    pid = int(pidfile.read_text())
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        state = "exited"
    summary["runner_pid"] = pid
    summary["runner_alive"] = state not in ("exited", "Z")
log = ROOT / (args.phase + ".log")
if log.exists():
    with log.open("rb") as f:
        f.seek(max(0, log.stat().st_size - 16384))
        lines = f.read().decode(errors="replace").splitlines()
    summary["latest_log"] = lines[-1][:200] if lines else None
for variant in ["dspark", "dspark_latentspec"]:
    file = ROOT / args.phase / variant / "results.jsonl"
    rows = []
    if file.exists():
        for line in file.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A concurrent append will be visible on the next read.
    valid = {r["case_id"]: r for r in rows if r["valid"]}
    summary["variants"][variant] = {
        "valid_cells": len(valid), "invalid_attempts": sum(not r["valid"] for r in rows),
        "policies": dict(Counter("natural512" if r["output_length"] == 0 else "fixed" + str(r["output_length"])
                                 for r in valid.values())),
        "last_case": rows[-1]["case_id"] if rows else None,
    }
summary["valid_cells"] = sum(r["valid_cells"] for r in summary["variants"].values())
print(json.dumps(summary, indent=2))
