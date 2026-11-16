#!/usr/bin/env python3
"""Read-only export for the explicitly ordered full-dataset test plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from full_common import ROOT, COUNTS, valid_cell, inherited_rows
from priority_plan import make_plan, selected_units, coordinates, key, versions, ack_digest, expected_tokens


def result_rows(out):
    manifest_path=out/"suite-manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    inherited=inherited_rows(out)
    plan=manifest["plan"] if manifest else make_plan()
    scope=set().union(*(coordinates(u) for u in plan["units"]))
    unit_coordinates={u["unit_id"]:coordinates(u) for u in plan["units"]}
    rows={key(r):dict(r,decode_cuda_graph_max_bs=32,measurement_source="full_v1")
          for r in inherited if key(r) in scope}
    for path in sorted((out/"measurements").glob("*/results.jsonl")):
        config_path=path.parent/"benchmark-config.json"
        if path.parent.name not in unit_coordinates or not config_path.exists():
            continue
        config=json.loads(config_path.read_text())
        fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        for line in path.read_text().splitlines():
            try:
                row=json.loads(line)
            except ValueError:
                continue
            name=row.get("dataset")
            if name not in COUNTS or not valid_cell(row,COUNTS[name]):
                continue
            if manifest and row.get("dataset_sha256") != manifest["datasets"][name]["sha256"]:
                continue
            if manifest and manifest.get("request_statistics_required"):
                statistics=row.get("output_statistics",{})
                if (statistics.get("successful_requests") != COUNTS[name]
                        or statistics.get("unknown_finish_reason_requests") != 0
                        or statistics.get("truncation_rate_percent") is None):
                    continue
            if (key(row) not in unit_coordinates[path.parent.name] or row.get("decode_cuda_graph_max_bs") != 256
                    or row.get("configuration_sha256") != fingerprint):
                continue
            compact={k:v for k,v in row.items() if not k.startswith("server_info")}
            compact["measurement_source"]=out.name
            rows[key(row)]=compact
    return list(rows.values())


def export(out):
    plan_path=out/"execution-plan.json"
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else make_plan()
    control_path=out/"execution-control.json"
    control=json.loads(control_path.read_text()) if control_path.exists() else {"stop_after_stage":None}
    units=selected_units(plan,control)
    scope=set().union(*(coordinates(u) for u in units))
    rows=[r for r in result_rows(out) if key(r) in scope]
    unit_status=[]
    for unit in units:
        expected=coordinates(unit)
        measured=[r for r in rows if key(r) in expected]
        complete=len(measured)==unit["expected_cells"]
        unit_status.append(dict(unit_id=unit["unit_id"],stage=unit["stage"],family=unit["family"],variant=unit["variant"],expected_cells=unit["expected_cells"],
            complete=complete,versions=versions(measured),sha256=ack_digest(measured) if complete else None))
    status_path=out/"status.json"
    status=json.loads(status_path.read_text()) if status_path.exists() else {"state":"preparing"}
    pid_path=out/"runner.pid"
    alive=False
    if pid_path.exists():
        try:
            args=Path('/proc')/pid_path.read_text().strip()/'cmdline'
            alive=b'priority_suite.py' in args.read_bytes()
        except FileNotFoundError:
            pass
    outcome_path=out/"measurements-outcome.json"
    outcome=json.loads(outcome_path.read_text()) if outcome_path.exists() else None
    progress=[]
    for path in (out/"measurements").glob("*/request-progress.json"):
        progress.append(json.loads(path.read_text()))
    return dict(rows=rows,completed_cells=len(rows),total_cells=len(scope),priority_cells=sum(u["expected_cells"] for u in units if u["stage"]<=4),
                expected_completion_tokens=expected_tokens(units),status=status,runner_alive=alive,progress=progress,
                units=unit_status,outcome=outcome,control=control)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"rebuttal/output/priority_v2")
    print(json.dumps(export(parser.parse_args().output),ensure_ascii=False))
