#!/usr/bin/env python3
"""Own server process groups, run/resume both variants sequentially on one GPU."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def serve_and_test(variant, args, extra):
    run_dir = args.output / args.phase / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError(f"Port {args.port} already has a server; refusing to replace it")
    with (run_dir / "server.log").open("a") as log:
        env = dict(os.environ, PORT=str(args.port))
        manifest = {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True),
            "environment": {k: v for k, v in env.items() if k.startswith(("SGLANG_", "TARGET_", "DRAFT_")) or k in ("MODEL_CACHE", "USE_MODEL_CACHE")},
            "benchmark_args": extra,
            "server_args": args.server_args_json,
            "launch_script_sha256": hashlib.sha256((ROOT / "rebuttal/launch_server.sh").read_bytes()).hexdigest(),
            "benchmark_script_sha256": hashlib.sha256((ROOT / "rebuttal/benchmark.py").read_bytes()).hexdigest(),
            "python": sys.version,
        }
        manifest["packages"] = {}
        for package in ["torch", "sglang", "triton", "aiohttp"]:
            try:
                manifest["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                manifest["packages"][package] = None
        manifest_path = run_dir / "manifest.json"
        patch_path = run_dir / "framework.patch"
        patch = subprocess.check_output(["git", "diff", "--", "python/sglang"], cwd=ROOT, text=True)
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "python/sglang"],
            cwd=ROOT, text=True,
        ).splitlines()
        for path in untracked:
            addition = subprocess.run(
                ["git", "diff", "--no-index", "--", "/dev/null", path],
                cwd=ROOT, text=True, capture_output=True,
            )
            if addition.returncode not in (0, 1):
                raise RuntimeError(addition.stderr)
            patch += addition.stdout
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume configuration changed; use a new --phase")
        if patch_path.exists() and patch_path.read_text() != patch:
            raise ValueError("Resume framework changed; use a new --phase")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        patch_path.write_text(patch)
        server = subprocess.Popen([str(ROOT / "rebuttal/launch_server.sh"), variant, *args.server_args_json],
                                  env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (run_dir / "server.pid").write_text(str(server.pid))
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited with {server.returncode}; see {run_dir}/server.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(3)
            else:
                raise TimeoutError("Server startup exceeded 900 seconds")
            cmd = [sys.executable, str(ROOT / "rebuttal/benchmark.py"), "--variant", variant,
                   "--phase", args.phase, "--port", str(args.port), "--output", str(run_dir), *extra]
            subprocess.run(cmd, check=True)
            if args.profile and variant == "dspark_latentspec":
                subprocess.run([sys.executable, str(ROOT / "rebuttal/profile_server.py"),
                                "--port", str(args.port), "--output", str(run_dir / "profiles")], check=True)
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
            except ProcessLookupError:
                pass
            time.sleep(3)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "rebuttal/output")
    p.add_argument("--phase", default="baseline")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--variants", nargs="+", default=["dspark", "dspark_latentspec"])
    p.add_argument("--server-args-json", type=json.loads, default=[], help="JSON string array passed directly to server")
    p.add_argument("--profile", action="store_true", help="Profile latent variant after benchmark, excluded from all metrics")
    args, extra = p.parse_known_args()
    if not isinstance(args.server_args_json, list) or not all(isinstance(v, str) for v in args.server_args_json):
        p.error("--server-args-json must be a string array")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "matrix.pid").write_text(str(os.getpid()))
    (args.output / args.phase).mkdir(parents=True, exist_ok=True)
    (args.output / args.phase / "runner.pid").write_text(str(os.getpid()))
    for variant in args.variants:
        serve_and_test(variant, args, extra)
    (args.output / (args.phase + ".complete.json")).write_text(json.dumps({"variants": args.variants, "args": extra}))
    print("MATRIX_COMPLETE", flush=True)
