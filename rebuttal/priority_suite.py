#!/usr/bin/env python3
"""Execute tps.md's priority stages; require verified publication between units."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from full_common import ROOT, DATASETS, COUNTS, CONCURRENCIES, atomic_json, valid_cell, inherited_rows, read_rows
from full_suite import TARGETS, STEPS, stage, sha, stop_server, compress_owned_log, framework_patch
from priority_plan import make_plan, coordinates, key, selected_units, ack_digest, expected_tokens


def status(out, state, **details):
    atomic_json(out / "status.json", dict(state=state, runner_pid=os.getpid(),
                updated_at=datetime.now(timezone.utc).isoformat(), **details))


def load_control(out):
    p = out / "execution-control.json"
    return json.loads(p.read_text()) if p.exists() else {"stop_after_stage": None}


def freeze(out, selection_plan=None):
    # The statistics revision explicitly cancels all stages after2.
    plan = json.loads(selection_plan.read_text()) if selection_plan else make_plan(max_stage=2)
    seen=set()
    for unit in plan["units"]:
        if (unit["family"] not in TARGETS or unit["variant"] not in ["dspark","dspark_latentspec","dflash","dflash-3L","dflash-4L","eagle3","latentspec"]
                or not unit["datasets"] or not set(unit["datasets"])<=set(DATASETS)
                or not unit["lengths"] or not set(unit["lengths"])<={512,1024,2048,4096}):
            raise ValueError("Invalid explicitly selected full-data test unit")
        points=coordinates(unit)
        if seen & points or len(points)!=unit["expected_cells"]:
            raise ValueError("Duplicate or inconsistent selected test cells")
        seen.update(points)
    if len(seen)!=plan["total_cells"] or plan["decode_cuda_graph_max_bs"]!=256:
        raise ValueError("Selected plan size or graph configuration is inconsistent")
    corpus = out / "datasets"
    corpus.mkdir(exist_ok=True)
    for name in DATASETS:
        target = corpus / (name + ".jsonl")
        if not target.exists():
            target.write_bytes((ROOT / "datasets" / target.name).read_bytes())
    scripts = ["priority_suite.py", "priority_plan.py", "export_priority.py", "full_benchmark.py",
               "full_common.py", "full_suite.py", "benchmark.py", "launch_full_server.sh", "request_statistics.py"]
    patch = framework_patch()
    manifest = dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        framework_patch_sha256=__import__('hashlib').sha256(patch.encode()).hexdigest(),
        scripts={name: sha(ROOT / "rebuttal" / name) for name in scripts},
        reference_sha256=sha(ROOT / "latentspec_scripts/test_tps.py"),
        plan=plan, datasets={d: {"sha256": sha(corpus / (d + ".jsonl")), "rows": COUNTS[d]} for d in DATASETS},
        inherited_results_sha256=sha(out / "inherited-results.json"),
        gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True),
        packages={p: importlib.metadata.version(p) for p in ["torch", "sglang", "triton", "aiohttp", "flashinfer-python", "transformers", "safetensors", "numpy"]},
        environment={k:v for k,v in os.environ.items() if k.startswith("SGLANG_")},
        protocol="Full rows per selected dataset; actual completion tokens / whole measured group wall; fixed length; warmups excluded",
        decode_cuda_graph_max_bs=256, max_running_requests=256, request_statistics_required=True)
    path = out / "suite-manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("Frozen source, plan, environment or datasets changed")
    atomic_json(path, manifest)
    atomic_json(out / "execution-plan.json", plan)
    (out / "framework.patch").write_text(patch)
    preserved = inherited_rows(out)
    if any(r["dataset_sha256"] != manifest["datasets"][r["dataset"]]["sha256"] for r in preserved):
        raise RuntimeError("Preserved results use a different dataset snapshot")
    return plan, corpus


def unit_rows(out, unit):
    # The exporter validates and deduplicates complete cell records.
    from export_priority import result_rows
    expected = coordinates(unit)
    return [r for r in result_rows(out) if key(r) in expected]


def measure(out, unit, corpus, port):
    folder = out / "measurements" / unit["unit_id"]
    folder.mkdir(parents=True, exist_ok=True)
    if len(unit_rows(out, unit)) == unit["expected_cells"]:
        print("SKIP_COMPLETE_UNIT " + unit["unit_id"], flush=True)
        return
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError("Benchmark port is occupied; leaving its server untouched")
    family, variant = unit["family"], unit["variant"]
    cache = Path("/dev/shm/rebuttal-full-models")
    status(out, "staging_models", **unit)
    target = stage(TARGETS[family], cache / family / "target")
    draft = stage(f"/mydata/models/{family}/{variant}/{STEPS[family]}", cache / "current-draft")
    if variant=="dflash-4L" and json.loads((cache/"current-draft/config.json").read_text()).get("num_hidden_layers")!=4:
        raise RuntimeError("DFlash-4L checkpoint does not contain four draft layers")
    identity = dict(family=family, variant=variant,
                    target={k:target[k] for k in ["source","files"]},
                    draft={k:draft[k] for k in ["source","files"]})
    previous = json.loads((out / "inherited-results.json").read_text()).get("models", {}).get(family + "/" + variant)
    if previous is not None and previous != identity:
        raise RuntimeError("Preserved results used different model checkpoint bytes")
    identity_path = folder / "models.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("Model checkpoint identity changed")
    atomic_json(identity_path, identity)
    env = dict(os.environ, TARGET_MODEL=str(cache/family/"target"), DRAFT_MODEL=str(cache/"current-draft"),
               MODEL_FAMILY=family, PORT=str(port))
    log_path = folder / "server.log"
    with log_path.open("a") as log:
        server = subprocess.Popen(["bash", str(ROOT/"rebuttal/launch_full_server.sh"), variant],
                                  cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status(out, "server_starting", server_pid=server.pid, **unit)
            deadline = time.monotonic() + 1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}; inspect {log_path}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/get_server_info", timeout=3) as r:
                        info = json.load(r)
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server startup exceeded 1200 seconds")
                    time.sleep(3)
            args = info.get("server_args", info)
            graph = args["cuda_graph_config"]["decode"]
            if graph["max_bs"] != 256 or 256 not in graph["bs"] or args["max_running_requests"] != 256:
                raise RuntimeError("Actual server configuration does not capture batch256")
            atomic_json(folder/"server-info.json", info)
            status(out, "benchmarking", server_pid=server.pid, **unit)
            command = [sys.executable, str(ROOT/"rebuttal/full_benchmark.py"), "--suite", out.name,
                       "--family", family, "--variant", variant, "--port", str(port), "--output", str(folder),
                       "--dataset-dir", str(corpus), "--requests", "0", "--datasets", *unit["datasets"],
                       "--lengths", *map(str,unit["lengths"]), "--inherited-results", str(out/"inherited-results.json")]
            client = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
            try:
                while True:
                    try:
                        code = client.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        if server.poll() is not None:
                            raise RuntimeError("Server exited during measurement")
                if code:
                    raise subprocess.CalledProcessError(code, command)
            finally:
                if client.poll() is None:
                    stop_server(client)
        finally:
            stop_server(server)
    compress_owned_log(log_path)
    if len(unit_rows(out, unit)) != unit["expected_cells"]:
        raise RuntimeError("Unit ended without all required valid full-data cells")


def main(args):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/"runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out/"runner.pid").write_text(str(os.getpid()))
    plan, corpus = freeze(out,args.plan)
    prefetch = out/"prefetch.json"
    status(out,"waiting_for_model_prefetch")
    prefetch_pid = out/"prefetch.pid"
    if prefetch_pid.exists():
        while not prefetch.exists():
            try:
                os.kill(int(prefetch_pid.read_text()),0)
            except ProcessLookupError:
                raise RuntimeError("Model prefetch exited without a result")
            time.sleep(5)
        if json.loads(prefetch.read_text())["state"] != "ready":
            raise RuntimeError(prefetch.read_text())
    for unit in plan["units"]:
        if unit not in selected_units(plan,load_control(out)):
            continue
        for attempt in [1,2]:
            try:
                print("UNIT " + json.dumps(unit), flush=True)
                measure(out,unit,corpus,args.port)
                break
            except Exception as exc:
                event=dict(unit_id=unit["unit_id"],attempt=attempt,error=str(exc),time=datetime.now(timezone.utc).isoformat())
                with (out/"errors.jsonl").open("a") as f:
                    f.write(json.dumps(event)+"\n")
                if attempt==2:
                    status(out,"failed",**event)
                    raise
                time.sleep(15)
        rows = unit_rows(out,unit)
        digest = ack_digest(rows)
        status(out,"awaiting_document_verification",**unit)
        ack = out/"publication-acks"/(unit["unit_id"]+".json")
        while not ack.exists() or json.loads(ack.read_text()).get("sha256") != digest:
            time.sleep(5)
        status(out,"unit_published",**unit)
        print("PUBLISHED " + unit["unit_id"],flush=True)
    chosen = selected_units(plan,load_control(out))
    from export_priority import result_rows
    scope=set().union(*(coordinates(u) for u in chosen))
    actual={key(r) for r in result_rows(out)}
    if not scope<=actual:
        raise RuntimeError("Selected plan has missing measurements")
    atomic_json(out/"measurements-outcome.json",dict(complete=True,total_cells=len(scope),
        total_completion_tokens=expected_tokens(chosen),finished_at=datetime.now(timezone.utc).isoformat()))
    status(out,"complete",total_cells=len(scope))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"rebuttal/output/priority_v2")
    parser.add_argument("--port",type=int,default=30000)
    parser.add_argument("--plan",type=Path,help="Explicit full-dataset selection plan for a new benchmark phase")
    main(parser.parse_args())
