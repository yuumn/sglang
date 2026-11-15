"""Shared full-suite definitions; standard library only for local publishing."""
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["math500", "aime25", "humaneval", "mbpp", "mt-bench", "alpaca"]
COUNTS = dict(zip(DATASETS, [500, 30, 164, 256, 80, 500]))
VARIANTS = ["dflash", "dflash-3L", "dspark", "dspark_latentspec", "eagle3", "latentspec"]
CONCURRENCIES = [1, 2, 4, 8, 16]
LENGTHS = [512, 1024, 2048, 4096]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def valid_cell(row, expected_count):
    length = row.get("output_length")
    return (row.get("valid") is True and row.get("full_dataset") is True
            and row.get("total_requests") == expected_count
            and row.get("successful_requests") == expected_count
            and row.get("failed_requests") == 0
            and length in LENGTHS
            and row.get("completion_tokens_min") == length
            and row.get("completion_tokens_max") == length
            and row.get("total_completion_tokens") == expected_count * length
            and row.get("wall_time_seconds", 0) > 0
            and math.isfinite(row.get("output_tokens_per_second", float("nan")))
            and math.isclose(row["output_tokens_per_second"],
                             row["total_completion_tokens"] / row["wall_time_seconds"], rel_tol=1e-12))


def read_rows(path):
    if not path.exists():
        return []
    # A process interruption may leave only the last append incomplete. Keep all
    # complete rows; discard that final torn write before the next append.
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        end = data.rfind(b"\n") + 1
        path.with_suffix(path.suffix + ".interrupted-tail").write_bytes(data[end:])
        with path.open("r+b") as f:
            f.truncate(end)
        data = data[:end]
    return [json.loads(line) for line in data.splitlines()]


