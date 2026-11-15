"""Shared full-suite definitions; standard library only for local publishing."""
import json
import hashlib
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["math500", "aime25", "humaneval", "mbpp", "mt-bench", "alpaca"]
COUNTS = dict(zip(DATASETS, [500, 30, 164, 256, 80, 500]))
VARIANTS = ["dflash", "dflash-3L", "dspark", "dspark_latentspec", "eagle3", "latentspec"]
CONCURRENCIES = [1, 2, 4, 8, 16]
LENGTHS = [512, 1024, 2048, 4096]


def inherited_rows(output):
    """Load explicitly preserved measurements without rewriting their provenance."""
    path = Path(output) / "inherited-results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    manifest_path = Path(output) / "suite-manifest.json"
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text()).get("inherited_results_sha256")
        if expected != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Inherited measurement checkpoint changed")
    seen = set()
    for row in data["rows"]:
        name = row.get("dataset")
        if name not in COUNTS or not valid_cell(row, COUNTS[name]):
            raise ValueError("Inherited row is not a complete full-dataset measurement")
        if row.get("dataset_sha256") != data["source_manifest"]["datasets"][name]["sha256"]:
            raise ValueError("Inherited dataset hash differs from its source manifest")
        if row.get("variant") not in VARIANTS or row.get("family") not in ("qwen3-4b", "qwen3-8b") or row.get("concurrency") not in CONCURRENCIES:
            raise ValueError("Invalid inherited matrix coordinates")
        sampling = row.get("sampling", {})
        if any(sampling.get(k) != v for k, v in {
            "max_completion_tokens": row["output_length"], "temperature": 1.0,
            "seed": 980406, "ignore_eos": True, "enable_thinking": False,
            "top_p": 1.0, "top_k": -1, "min_p": 0.0,
            "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
            "stop": None, "extra_request_body": {},
        }.items()):
            raise ValueError("Inherited sampling settings differ")
        key = (row["family"], row["variant"], name, row["concurrency"], row["output_length"])
        if key in seen:
            raise ValueError("Duplicate inherited measurement")
        seen.add(key)
    return data["rows"]


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


