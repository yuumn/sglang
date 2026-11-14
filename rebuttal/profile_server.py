#!/usr/bin/env python3
"""Collect a short server-side CPU/GPU trace outside measured benchmark cells."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import urllib.request


def post(base, path, body):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read().decode()
        try:
            return json.loads(data)
        except ValueError:
            return data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as result:
        model = json.load(result)["data"][0]["id"]
    post(base, "/start_profile", {"output_dir": str(args.output.resolve()), "num_steps": 40,
                                  "activities": ["CPU", "GPU"], "record_shapes": True})
    response = post(base, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Explain how to solve a quadratic equation, with several examples."}],
        "max_completion_tokens": 1024, "temperature": 1.0, "seed": 980406,
        "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False},
    })
    (args.output / "profile-request.json").write_text(json.dumps(response))
    try:
        post(base, "/stop_profile", {})
    except Exception as exc:
        print("Profiler may already have stopped after 40 steps: " + str(exc))
    totals = defaultdict(lambda: {"calls": 0, "duration_us": 0.0})
    for file in [*args.output.glob("*.trace.json.gz"), *args.output.glob("*.trace.json")]:
        opener = gzip.open if file.suffix == ".gz" else open
        with opener(file, "rt") as f:
            trace = json.load(f)
        for event in trace.get("traceEvents", []):
            if "kernel" not in event.get("cat", "").lower() or "dur" not in event:
                continue
            item = totals[event["name"]]
            item["calls"] += 1
            item["duration_us"] += event["dur"]
    rows = sorted([{"name": name, **data} for name, data in totals.items()],
                  key=lambda x: -x["duration_us"])
    (args.output / "kernel-summary.json").write_text(json.dumps(rows, indent=2))
    for row in rows[:30]:
        print(json.dumps(row), flush=True)
