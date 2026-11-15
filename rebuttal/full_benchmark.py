#!/usr/bin/env python3
"""All-row fixed-length matrix with strict validation and live request progress.

An interrupted cell is rerun in its entirety; timings from separate attempts are
never combined. Only complete, valid cells can be skipped on resume.
"""
import argparse
import asyncio
from contextlib import suppress
from copy import copy
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import aiohttp
from benchmark import ROOT, append, info, reference

from full_common import DATASETS, COUNTS, VARIANTS, CONCURRENCIES, LENGTHS, atomic_json, valid_cell, read_rows


async def measured_requests(session, base, headers, model, prompts, concurrency,
                            sampling, progress_path, case_id):
    semaphore = asyncio.Semaphore(concurrency)
    finished = 0
    successful_tokens = 0
    failed = 0
    started = time.perf_counter()

    def report():
        atomic_json(progress_path, {
            "case_id": case_id, "finished_requests": finished,
            "total_requests": len(prompts), "failed_requests": failed,
            "successful_completion_tokens": successful_tokens,
            "elapsed_seconds": time.perf_counter() - started,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    async def heartbeat():
        while True:
            await asyncio.sleep(30)
            report()

    async def one(prompt):
        nonlocal finished, successful_tokens, failed
        async with semaphore:
            result = await reference.send_request(
                session=session, url=base + "/v1/chat/completions", headers=headers,
                body=reference.build_request_body(sampling, model, prompt))
            finished += 1
            successful_tokens += result.completion_tokens if result.success else 0
            failed += not result.success
            if not result.success:
                raise RuntimeError("Request failed; aborting whole cell: " + str(result.error))
            return result

    reporter = asyncio.create_task(heartbeat())
    tasks = [asyncio.create_task(one(prompt)) for prompt in prompts]
    try:
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - started
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        reporter.cancel()
        with suppress(asyncio.CancelledError):
            await reporter
        report()
    # Request completion does no synchronous disk I/O. Final telemetry is outside
    # the measured wall interval; periodic telemetry is at most once per 30s.
    return results, wall


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    prompts, metadata = reference.resolve_datasets(args.dataset_dir, DATASETS, args.requests)
    for name, details in metadata.items():
        details["sha256"] = hashlib.sha256((args.dataset_dir / (name + ".jsonl")).read_bytes()).hexdigest()
        if details["available_prompts"] != COUNTS[name]:
            raise ValueError(f"Unexpected corpus size for {name}: {details}")
        if not args.smoke and details["selected_prompts"] != COUNTS[name]:
            raise ValueError("Final measurements must use every dataset row")
    configuration = {
        "family": args.family, "variant": args.variant, "datasets": metadata,
        "requests": args.requests, "concurrencies": args.concurrencies,
        "lengths": args.lengths, "temperature": 1.0, "sampling_seed": 980406,
        "full_dataset": not args.smoke,
    }
    config_path = args.output / "benchmark-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Resume settings changed; use a new suite directory")
    atomic_json(config_path, configuration)
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    results_file = args.output / "results.jsonl"
    rows = read_rows(results_file)
    read_rows(args.output / "requests.jsonl")  # Recover a torn per-request append too.
    completed = {r["case_id"] for r in rows
                 if valid_cell(r, COUNTS[r["dataset"]])
                 and r.get("configuration_sha256") == fingerprint}
    base = f"http://127.0.0.1:{args.port}"
    headers = {"Content-Type": "application/json"}
    sampling = argparse.Namespace(
        max_tokens=128, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
        frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0,
        ignore_eos=True, enable_thinking=False, sampling_seed=980406,
        stop=None, extra_request_body={})
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600),
                                     connector=aiohttp.TCPConnector(limit=0)) as session:
        model = await reference.discover_model(session, base, headers)
        append(args.output / "metadata.jsonl", {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "server_info": await info(session, base), "configuration": configuration,
            "primary_metric": "sum(actual completion tokens) / entire measured group wall seconds"})
        all_datasets = list(prompts.values())
        count = len(all_datasets)
        warmup = [all_datasets[i % count][(i // count) % len(all_datasets[i % count])] for i in range(16)]
        warmed, _ = await reference.run_requests(
            session=session, url=base + "/v1/chat/completions", headers=headers,
            model=model, prompts=warmup, concurrency=16, args=sampling)
        if any(not r.success for r in warmed):
            raise RuntimeError("Global warmup failed")
        for length in args.lengths:
            sampling.max_tokens = length
            for concurrency in args.concurrencies:
                for name, selected in prompts.items():
                    case_id = f"{args.suite}/{args.family}/{args.variant}/{name}/c{concurrency}/fixed{length}"
                    if case_id in completed:
                        continue
                    if len(selected) < concurrency:
                        raise ValueError("Selected prompts cannot reach requested concurrency")
                    await reference.flush_cache(session, base, headers, 60)
                    case_warmup = copy(sampling)
                    case_warmup.max_tokens = 128
                    warmed, _ = await reference.run_requests(
                        session=session, url=base + "/v1/chat/completions", headers=headers,
                        model=model, prompts=selected[:concurrency], concurrency=concurrency, args=case_warmup)
                    if any(not r.success for r in warmed):
                        raise RuntimeError("Case warmup failed")
                    await reference.flush_cache(session, base, headers, 60)
                    before = await info(session, base)
                    print("START " + case_id, flush=True)
                    atomic_json(args.output / "request-progress.json", {
                        "case_id": case_id, "finished_requests": 0, "total_requests": len(selected),
                        "updated_at": datetime.now(timezone.utc).isoformat()})
                    try:
                        results, wall = await measured_requests(
                            session, base, headers, model, selected, concurrency, sampling,
                            args.output / "request-progress.json", case_id)
                    except Exception as exc:
                        append(args.output / "failed-attempts.jsonl", {
                            "case_id": case_id, "error": str(exc),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "progress": json.loads((args.output / "request-progress.json").read_text()),
                            "valid": False, "partial_attempt_only": True})
                        raise
                    summary = reference.summarize_group(
                        dataset_name=name, concurrency=concurrency, results=results, wall_time_seconds=wall)
                    errors = [i for i, r in enumerate(results) if r.success and r.completion_tokens != length]
                    record = dict(
                        case_id=case_id, suite=args.suite, family=args.family, variant=args.variant,
                        created_at=datetime.now(timezone.utc).isoformat(), output_length=length,
                        full_dataset=not args.smoke, configuration_sha256=fingerprint, **summary,
                        dataset_sha256=metadata[name]["sha256"], case_warmup_requests=concurrency,
                        case_warmup_max_tokens=128, output_tokens_per_second=summary["total_completion_tokens"] / wall,
                        valid=not summary["failed_requests"] and not errors, length_error_indices=errors,
                        sampling=reference.sampling_config(sampling),
                        completion_tokens_min=min(r.completion_tokens for r in results),
                        completion_tokens_max=max(r.completion_tokens for r in results),
                        server_info_before=before, server_info_after=await info(session, base))
                    append(args.output / "requests.jsonl", {
                        "case_id": case_id, "created_at": record["created_at"],
                        "requests": [dataclasses.asdict(r) for r in results]})
                    append(results_file, record)
                    print("RESULT " + json.dumps({k: v for k, v in record.items()
                                                   if not k.startswith("server_info")}), flush=True)
                    if not record["valid"]:
                        raise RuntimeError("Invalid full-dataset measurement: " + case_id)
        atomic_json(args.output / "complete.json", {
            "configuration_sha256": fingerprint, "full_dataset": not args.smoke,
            "expected_cells": len(DATASETS) * len(args.lengths) * len(args.concurrencies),
            "finished_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="full_v1")
    p.add_argument("--family", required=True, choices=["qwen3-4b", "qwen3-8b"])
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--requests", type=int, default=0)
    p.add_argument("--concurrencies", nargs="+", type=int, default=CONCURRENCIES)
    p.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    p.add_argument("--smoke", action="store_true", help="Separate preflight only; never published as full data")
    args = p.parse_args()
    if args.requests != 0 and not args.smoke:
        p.error("Final benchmark requires --requests 0 (every row)")
    if not args.smoke and (args.concurrencies != CONCURRENCIES or args.lengths != LENGTHS):
        p.error("Final benchmark requires all five concurrencies and four fixed lengths")
    asyncio.run(main(args))
