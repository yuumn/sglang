#!/usr/bin/env python3
"""Emit compact records for the optional local Feishu synchronizer."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "output"
KEYS = ["case_id", "phase", "variant", "dataset", "concurrency", "output_length",
        "natural_max_tokens", "created_at", "valid", "output_tokens_per_second",
        "tps_excluding_first_token", "total_requests", "successful_requests", "failed_requests",
        "total_completion_tokens", "completion_tokens_min", "completion_tokens_max", "wall_time_seconds"]
for path in sorted(ROOT.glob("**/results.jsonl")):
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # A concurrent append will be read on the next pass.
        print(json.dumps({k: record.get(k) for k in KEYS}, ensure_ascii=False))
