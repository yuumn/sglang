#!/usr/bin/env python3
"""Matched streaming throughput matrix, reusing latentspec_scripts/test_tps.py."""
from __future__ import annotations

import argparse
import asyncio
from copy import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "latentspec_scripts"))
import test_tps as reference
import aiohttp


def append(path, obj):
    with Path(path).open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


async def info(session, base):
    async with session.get(base + "/get_server_info") as response:
        response.raise_for_status()
        return await response.json()


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    results_file = args.output / "results.jsonl"
    requests_file = args.output / "requests.jsonl"
    completed = set()
    if results_file.exists():
        completed = {x["case_id"] for line in results_file.read_text().splitlines()
                     if (x := json.loads(line)).get("valid")}
    prompts, metadata = reference.resolve_datasets(ROOT / "datasets", args.datasets, args.requests)
    for name in metadata:
        metadata[name]["sha256"] = hashlib.sha256((ROOT / "datasets" / (name + ".jsonl")).read_bytes()).hexdigest()
    configuration = {"datasets": metadata, "requests": args.requests,
                     "concurrencies": args.concurrencies, "lengths": args.lengths,
                     "natural_max_tokens": args.natural_max_tokens, "temperature": args.temperature}
    config_path = args.output / "benchmark-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Resume benchmark settings changed; use a new --phase")
    config_path.write_text(json.dumps(configuration, indent=2))
    base = f"http://127.0.0.1:{args.port}"
    headers = {"Content-Type": "application/json"}
    sampling = argparse.Namespace(
        max_tokens=128, temperature=args.temperature, top_p=1.0, top_k=-1, min_p=0.0,
        frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0,
        ignore_eos=True, enable_thinking=False, sampling_seed=980406,
        stop=None, extra_request_body={},
    )
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600),
                                     connector=aiohttp.TCPConnector(limit=0)) as session:
        model = await reference.discover_model(session, base, headers)
        environment = await info(session, base)
        append(args.output / "metadata.jsonl", {
            "variant": args.variant, "phase": args.phase, "datasets": metadata,
            "server_info": environment, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "primary_metric": "total successful completion tokens / entire measured group wall seconds",
            "fixed_length": "ignore_eos=true, no custom stop; every request length is checked",
        })
        all_datasets = list(prompts.values())
        warmup = [all_datasets[i % len(all_datasets)][(i // len(all_datasets)) % len(all_datasets[i % len(all_datasets)])]
                  for i in range(16)]
        warmed, _ = await reference.run_requests(
            session=session, url=base + "/v1/chat/completions", headers=headers,
            model=model, prompts=warmup, concurrency=16, args=sampling)
        if any(not r.success for r in warmed):
            raise RuntimeError("Warmup failed: " + str([r.error for r in warmed if not r.success]))
        for length in args.lengths:
            sampling.max_tokens = length or args.natural_max_tokens
            sampling.ignore_eos = length != 0
            for concurrency in args.concurrencies:
                for name, selected in prompts.items():
                    case_id = f"{args.phase}/{args.variant}/{name}/c{concurrency}/{'fixed'+str(length) if length else 'natural'+str(args.natural_max_tokens)}"
                    if case_id in completed:
                        continue
                    if len(selected) < concurrency:
                        raise ValueError(f"{case_id}: {len(selected)} prompts cannot reach concurrency {concurrency}")
                    await reference.flush_cache(session, base, headers, 60)
                    # Exclude first-use prefill/GEMM compilation for this dataset
                    # and batch shape. Clear its prefix cache before timing.
                    case_warmup = copy(sampling)
                    case_warmup.max_tokens = 128
                    case_warmup.ignore_eos = True
                    warmed, _ = await reference.run_requests(
                        session=session, url=base + "/v1/chat/completions", headers=headers,
                        model=model, prompts=selected[:concurrency], concurrency=concurrency, args=case_warmup)
                    if any(not r.success for r in warmed):
                        raise RuntimeError("Case warmup failed: " + str([r.error for r in warmed if not r.success]))
                    await reference.flush_cache(session, base, headers, 60)
                    before = await info(session, base)
                    print("START " + case_id, flush=True)
                    results, wall = await reference.run_requests(
                        session=session, url=base + "/v1/chat/completions", headers=headers,
                        model=model, prompts=selected, concurrency=concurrency, args=sampling)
                    after = await info(session, base)
                    summary = reference.summarize_group(dataset_name=name, concurrency=concurrency,
                                                         results=results, wall_time_seconds=wall)
                    length_errors = [i for i, r in enumerate(results) if r.success and length and r.completion_tokens != length]
                    record = dict(
                        case_id=case_id, phase=args.phase, variant=args.variant,
                        created_at=datetime.now(timezone.utc).isoformat(), output_length=length,
                        natural_max_tokens=args.natural_max_tokens, **summary,
                        case_warmup_requests=concurrency, case_warmup_max_tokens=128,
                        output_tokens_per_second=summary["total_completion_tokens"] / wall,
                        valid=not summary["failed_requests"] and not length_errors,
                        length_error_indices=length_errors, sampling=reference.sampling_config(sampling),
                        completion_tokens_min=min(r.completion_tokens for r in results),
                        completion_tokens_max=max(r.completion_tokens for r in results),
                        server_info_before=before, server_info_after=after,
                    )
                    append(requests_file, {"case_id": case_id, "requests": [dataclasses.asdict(r) for r in results]})
                    append(results_file, record)
                    completed.add(case_id)
                    print("RESULT " + json.dumps({k: v for k, v in record.items() if not k.startswith("server_info")}, ensure_ascii=False), flush=True)
                    if not record["valid"]:
                        raise RuntimeError("Invalid benchmark: " + case_id)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=["dspark", "dspark_latentspec"])
    p.add_argument("--phase", default="baseline")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--datasets", nargs="+", default=["math500", "aime25", "humaneval", "mbpp", "mt-bench", "alpaca"])
    p.add_argument("--concurrencies", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--lengths", nargs="+", type=int, default=[0, 512, 1024, 2048, 4096])
    p.add_argument("--natural-max-tokens", type=int, default=512)
    p.add_argument("--requests", type=int, default=32, help="Matched first N prompts, 0 = entire dataset")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if any(c < 1 for c in args.concurrencies) or any(n < 0 for n in args.lengths) or args.requests < 0:
        p.error("Invalid matrix dimensions")
    asyncio.run(main(args))
