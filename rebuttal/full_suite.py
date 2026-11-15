#!/usr/bin/env python3
"""Durable, single-GPU two-target/six-draft full benchmark orchestration.

Use --preflight first, then run without it. This process survives SSH disconnects
when started with nohup; each cell is fsynced and resumable. All failures are
recorded. Failed configurations do not prevent testing the other configurations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from full_common import ROOT, DATASETS, COUNTS, VARIANTS, atomic_json, read_rows, valid_cell, inherited_rows

TARGETS = {"qwen3-4b": "/mydata/Qwen/Qwen3-4B", "qwen3-8b": "/root/workspace/Qwen/Qwen3-8B"}
STEPS = {"qwen3-4b": "step_2616", "qwen3-8b": "step_2618"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def stage(source, destination):
    """Use only our marked cache directory; verify copied bytes before serving."""
    source = Path(source)
    manifest_path = destination / ".verified.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["source"] == str(source):
            try:
                intact = all((destination / name).stat().st_size == row["bytes"]
                             and sha(destination / name) == row["sha256"]
                             for name, row in previous["files"].items())
            except OSError:
                intact = False
            if intact:
                return previous
    if destination.exists():
        if not (destination / ".rebuttal-owned").exists():
            raise RuntimeError("Refusing to replace an unowned cache: " + str(destination))
        shutil.rmtree(destination)
    files = sorted(p for p in source.iterdir() if p.is_file())
    if not files or not (source / "config.json").exists():
        raise RuntimeError("Incomplete checkpoint: " + str(source))
    destination.mkdir(parents=True)
    (destination / ".rebuttal-owned").write_text("full_suite.py disposable verified copy\n")
    required = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(destination).free < required + 1024**3:
        raise RuntimeError(f"Insufficient shared memory for {source}: need {required} bytes + 1GiB reserve")

    def copy_file(src):
        print("STAGE " + str(src), flush=True)
        out = destination / src.name
        before = src.stat()
        digest = hashlib.sha256()
        with src.open("rb") as fi, out.open("wb") as fo:
            while chunk := fi.read(8 * 1024**2):
                fo.write(chunk)
                digest.update(chunk)
        after = src.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise RuntimeError("Source changed during copy: " + str(src))
        if out.stat().st_size != before.st_size or sha(out) != digest.hexdigest():
            raise RuntimeError("Checkpoint verification failed: " + str(src))
        return src.name, {"bytes": out.stat().st_size, "sha256": digest.hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = dict(pool.map(copy_file, files))
    manifest = {"source": str(source), "files": entries,
                "verified_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(manifest_path, manifest)
    return manifest


def framework_patch():
    patch = subprocess.check_output(["git", "diff", "--", "python/sglang"], cwd=ROOT, text=True)
    for path in subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "python/sglang"],
            cwd=ROOT, text=True).splitlines():
        done = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", path],
                              cwd=ROOT, text=True, capture_output=True)
        if done.returncode not in (0, 1):
            raise RuntimeError(done.stderr)
        patch += done.stdout
    return patch


def freeze_inputs(output, preflight_source=None):
    corpus = output / "datasets"
    corpus.mkdir(exist_ok=True)
    dataset_manifest = {}
    for name in DATASETS:
        source = ROOT / "datasets" / (name + ".jsonl")
        dest = corpus / source.name
        if not dest.exists():
            shutil.copyfile(source, dest)
        dataset_manifest[name] = {"sha256": sha(dest), "rows": COUNTS[name]}
    scripts = ["rebuttal/full_suite.py", "rebuttal/full_benchmark.py", "rebuttal/full_common.py", "rebuttal/launch_full_server.sh",
               "rebuttal/benchmark.py", "latentspec_scripts/test_tps.py"]
    patch = framework_patch()
    manifest = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "framework_patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "scripts": {p: sha(ROOT / p) for p in scripts}, "datasets": dataset_manifest,
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True),
        "python": sys.version,
        "packages": {p: importlib.metadata.version(p) for p in [
            "torch", "sglang", "triton", "aiohttp", "flashinfer-python",
            "transformers", "safetensors", "numpy"]},
        "targets": TARGETS, "variants": VARIANTS,
        "sampling": {"temperature": 1.0, "seed": 980406, "enable_thinking": False, "ignore_eos": True},
        "protocol": "FULL_DATASETS; total completion tokens / whole measured group wall; one attempt per cell",
        "decode_cuda_graph_max_bs": 256,
        "max_running_requests": 256,
        "preflight_reference": str(preflight_source) if preflight_source else None,
        "inherited_results_sha256": sha(output / "inherited-results.json") if (output / "inherited-results.json").exists() else None,
        "environment": {k: v for k, v in os.environ.items() if k.startswith("SGLANG_")},
    }
    path = output / "suite-manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("Frozen code, datasets, hardware or environment changed; use a new suite directory")
    atomic_json(path, manifest)
    (output / "framework.patch").write_text(patch)
    return corpus


def fully_complete(run_dir, family, variant):
    rows = read_rows(run_dir / "results.jsonl") + inherited_rows(run_dir.parents[2])
    expected = {(d, c, n) for d in DATASETS for c in [1, 2, 4, 8, 16] for n in [512, 1024, 2048, 4096]}
    actual = {(r["dataset"], r["concurrency"], r["output_length"]) for r in rows
              if r.get("family") == family and r.get("variant") == variant
              and valid_cell(r, COUNTS.get(r.get("dataset"), -1))}
    return (run_dir / "complete.json").exists() and actual == expected


def stop_server(server):
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()
    except ProcessLookupError:
        pass


def compress_owned_log(path):
    if path.exists():
        destination = path.with_suffix(path.suffix + ".gz")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if destination.exists():
            shutil.copyfile(destination, temporary)
        else:
            temporary.write_bytes(b"")
        # Append a gzip member on repeated attempts, preserving earlier logs.
        with path.open("rb") as src, temporary.open("ab") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=3) as dest:
                shutil.copyfileobj(src, dest)
        temporary.replace(destination)
        path.unlink()


def run_configuration(args, family, variant, corpus):
    mode = "preflight" if args.preflight else "measurements"
    run_dir = args.output / mode / family / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.preflight and fully_complete(run_dir, family, variant):
        print(f"SKIP_FULLY_COMPLETE {family}/{variant}", flush=True)
        return
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError("Benchmark port already serves another process; refusing to replace it")
    target_path = args.cache / family / "target"
    draft_path = args.cache / "current-draft"
    target = stage(TARGETS[family], target_path)
    draft = stage(f"/mydata/models/{family}/{variant}/{STEPS[family]}", draft_path)
    identity = {"family": family, "variant": variant,
                "target": {"source": target["source"], "files": target["files"]},
                "draft": {"source": draft["source"], "files": draft["files"]}}
    identity_path = run_dir / "models.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("Checkpoint identity changed; use a new suite directory")
    atomic_json(identity_path, identity)
    env = dict(os.environ, TARGET_MODEL=str(target_path), DRAFT_MODEL=str(draft_path),
               MODEL_FAMILY=family, PORT=str(args.port))
    server_log = run_dir / "server.log"
    with server_log.open("a") as log:
        server = subprocess.Popen(["bash", str(ROOT / "rebuttal/launch_full_server.sh"), variant],
                                  env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(args.output / "status.json", {
            "runner_pid": os.getpid(), "server_pid": server.pid, "mode": mode,
            "family": family, "variant": variant, "state": "server_starting",
            "updated_at": datetime.now(timezone.utc).isoformat()})
        try:
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited with {server.returncode}: {server_log}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(3)
            else:
                raise TimeoutError("Server startup exceeded 1200 seconds")
            atomic_json(args.output / "status.json", {
                "runner_pid": os.getpid(), "server_pid": server.pid, "mode": mode,
                "family": family, "variant": variant, "state": "benchmarking",
                "updated_at": datetime.now(timezone.utc).isoformat()})
            cmd = [sys.executable, str(ROOT / "rebuttal/full_benchmark.py"),
                   "--suite", args.output.name, "--family", family, "--variant", variant,
                   "--dataset-dir", str(corpus), "--output", str(run_dir), "--port", str(args.port)]
            if args.preflight:
                cmd += ["--smoke", "--requests", "16", "--lengths", "512", "--concurrencies", "1", "16"]
            elif (args.output / "inherited-results.json").exists():
                cmd += ["--inherited-results", str(args.output / "inherited-results.json")]
            client = subprocess.Popen(cmd, start_new_session=True)
            try:
                while True:
                    try:
                        code = client.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        if server.poll() is not None:
                            raise RuntimeError("Server exited while benchmark client was running")
                if code:
                    raise subprocess.CalledProcessError(code, cmd)
            except BaseException:
                if client.poll() is None:
                    stop_server(client)
                raise
        finally:
            stop_server(server)
            time.sleep(3)
    compress_owned_log(server_log)


def main(args):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}; stopping owned server")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.output / "runner.pid").write_text(str(os.getpid()))
    corpus = freeze_inputs(args.output, args.preflight_source)
    inherited_rows(args.output)
    failures = []
    for family in args.families:
        for variant in args.variants:
            if not args.preflight:
                preflight = (args.preflight_source or args.output) / "preflight" / family / variant / "complete.json"
                if not preflight.exists():
                    raise RuntimeError(f"Run --preflight first: missing {preflight}")
            for attempt in range(1, args.attempts + 1):
                try:
                    print(f"CONFIGURATION {family}/{variant} attempt {attempt}", flush=True)
                    run_configuration(args, family, variant, corpus)
                    break
                except Exception as exc:
                    event = {"family": family, "variant": variant, "attempt": attempt,
                             "error": str(exc), "time": datetime.now(timezone.utc).isoformat()}
                    with (args.output / "errors.jsonl").open("a") as f:
                        f.write(json.dumps(event) + "\n")
                    print("CONFIGURATION_ERROR " + json.dumps(event), flush=True)
                    run_dir = args.output / ("preflight" if args.preflight else "measurements") / family / variant
                    compress_owned_log(run_dir / "server.log")
                    if attempt == args.attempts:
                        failures.append(event)
                    else:
                        time.sleep(15)
    mode = "preflight" if args.preflight else "measurements"
    expected_all = args.families == list(TARGETS) and args.variants == VARIANTS
    completed = not failures and (args.preflight or all(
        fully_complete(args.output / mode / family / variant, family, variant)
        for family in args.families for variant in args.variants))
    atomic_json(args.output / (mode + "-outcome.json"), {
        "complete": completed, "all_requested_configurations": expected_all,
        "families": args.families, "variants": args.variants,
        "failures": failures, "finished_at": datetime.now(timezone.utc).isoformat()})
    atomic_json(args.output / "status.json", {
        "runner_pid": os.getpid(), "state": "complete" if completed else "failed",
        "mode": mode, "all_requested_configurations": expected_all,
        "updated_at": datetime.now(timezone.utc).isoformat()})
    if not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "rebuttal/output/full_v1")
    p.add_argument("--cache", type=Path, default=Path("/dev/shm/rebuttal-full-models"))
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--preflight-source", type=Path, help="Reuse recorded model compatibility checks; new servers still capture graphs and warm up")
    p.add_argument("--families", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    p.add_argument("--attempts", type=int, default=2)
    main(p.parse_args())
