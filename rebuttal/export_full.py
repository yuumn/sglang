#!/usr/bin/env python3
"""Read-only compact export of actual all-row cells and live progress."""
import argparse
import json
import os
from pathlib import Path
from full_common import ROOT, DATASETS, COUNTS, VARIANTS, valid_cell


def export(output):
    manifest_path = output / "suite-manifest.json"
    if not manifest_path.exists():
        return {"rows": [], "total_cells": 1440, "state": "not_started"}
    manifest = json.loads(manifest_path.read_text())
    rows = {}
    progress = []
    for path in sorted((output / "measurements").glob("*/*/results.jsonl")):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                # The writer may be appending its last line concurrently.
                continue
            name = row.get("dataset")
            if (name not in COUNTS or not valid_cell(row, COUNTS[name])
                    or row.get("dataset_sha256") != manifest["datasets"][name]["sha256"]):
                continue
            if row.get("family") not in manifest["targets"] or row.get("variant") not in VARIANTS:
                continue
            key = (row["family"], row["variant"], name, row["concurrency"], row["output_length"])
            rows[key] = {k: v for k, v in row.items() if not k.startswith("server_info")}
    for path in sorted((output / "measurements").glob("*/*/request-progress.json")):
        progress.append(json.loads(path.read_text()))
    status = json.loads((output / "status.json").read_text()) if (output / "status.json").exists() else {}
    pid_file = output / "runner.pid"
    alive = False
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), 0)
            alive = True
        except ProcessLookupError:
            pass
    outcome_path = output / "measurements-outcome.json"
    outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else None
    preflight_cells = set()
    for path in sorted((output / "preflight").glob("*/*/results.jsonl")):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("valid") is True and row.get("full_dataset") is False:
                preflight_cells.add(row["case_id"])
    preflight_configs = len(list((output / "preflight").glob("*/*/complete.json")))
    if alive and (not status or status.get("runner_pid") != int(pid_file.read_text())):
        command = Path(f"/proc/{int(pid_file.read_text())}/cmdline").read_bytes()
        status = {"state": "staging_models", "mode": "preflight" if b"--preflight" in command else "measurements"}
    return {"rows": list(rows.values()), "completed_cells": len(rows), "total_cells": 1440,
            "status": status, "runner_alive": alive, "progress": progress,
            "preflight_valid_cells": len(preflight_cells), "preflight_completed_configurations": preflight_configs,
            "outcome": outcome, "manifest": manifest}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "rebuttal/output/full_v1")
    args = p.parse_args()
    print(json.dumps(export(args.output), ensure_ascii=False))
