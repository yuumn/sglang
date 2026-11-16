"""The requested four priority stages, followed by the remaining full matrix."""
import hashlib
import json

from full_common import DATASETS, COUNTS, CONCURRENCIES, VARIANTS

LENGTHS = [512, 2048, 4096]


def key(row):
    return "/".join(map(str, (row["family"], row["variant"], row["dataset"], row["concurrency"], row["output_length"])))


def coordinates(unit):
    return {"/".join(map(str, (unit["family"], unit["variant"], d, c, n)))
            for d in unit["datasets"] for c in CONCURRENCIES for n in unit["lengths"]}


def make_plan(max_stage=None):
    units = []

    def add(stage, family, variant, datasets, lengths):
        units.append(dict(unit_id=f"u{len(units)+1:02d}_{family}_{variant}", stage=stage,
                          family=family, variant=variant, datasets=datasets, lengths=lengths,
                          expected_cells=len(datasets) * 5 * len(lengths)))

    for stage, family in [(1, "qwen3-4b"), (2, "qwen3-8b")]:
        for variant in ["dspark", "dspark_latentspec"]:
            add(stage, family, variant, DATASETS[:2], LENGTHS)
    for stage, family in [(3, "qwen3-4b"), (4, "qwen3-8b")]:
        for variant in ["latentspec", "dflash"]:
            add(stage, family, variant, DATASETS[2:], [512])
    seen = set().union(*(coordinates(u) for u in units))
    for family in ["qwen3-4b", "qwen3-8b"]:
        for variant in VARIANTS:
            for length in LENGTHS:
                datasets = [d for d in DATASETS
                            if "/".join(map(str, (family, variant, d, 1, length))) not in seen]
                if datasets:
                    add(5, family, variant, datasets, [length])
                    seen.update(coordinates(units[-1]))
    if max_stage is not None:
        units=[u for u in units if u["stage"]<=max_stage]
    return {"version": 2, "decode_cuda_graph_max_bs": 256, "concurrencies": CONCURRENCIES,
            "lengths": LENGTHS, "priority_cells": sum(u["expected_cells"] for u in units if u["stage"]<=4),
            "total_cells": sum(u["expected_cells"] for u in units), "units": units}


def selected_units(plan, control):
    ceiling = control.get("stop_after_stage")
    return [u for u in plan["units"] if ceiling is None or u["stage"] <= ceiling]


def versions(rows):
    return {key(r): hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in rows}


def ack_digest(rows):
    return hashlib.sha256(json.dumps(versions(rows), sort_keys=True).encode()).hexdigest()


def expected_tokens(units):
    return sum(sum(COUNTS[d] for d in u["datasets"]) * 5 * sum(u["lengths"]) for u in units)
