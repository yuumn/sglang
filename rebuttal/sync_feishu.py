#!/usr/bin/env python3
"""Run on the authenticated local machine; publish each remote result once."""
import argparse
from html import escape
import json
from pathlib import Path
import subprocess
import time

DOC = "RYK9dybIToz94AxBtZlcHsDTngv"


def lark(*args):
    done = subprocess.run(["lark-cli", "docs", *args, "--as", "user"], capture_output=True, text=True, timeout=120)
    try:
        payload = json.loads(done.stdout)
    except ValueError:
        raise RuntimeError(done.stderr[:1000] or done.stdout[:1000])
    if done.returncode or not payload.get("ok"):
        raise RuntimeError(json.dumps(payload, ensure_ascii=False))
    return payload


def sync(args):
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    done = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "h20",
                           "/root/workspace/sglang/.venv/bin/python /root/workspace/sglang/rebuttal/export_results.py"],
                          check=True, capture_output=True, text=True, timeout=60)
    for line in done.stdout.splitlines():
        row = json.loads(line)
        key = row["case_id"]
        version_key = key + "@" + row["created_at"]
        if version_key in state or state.get(key) == row:
            continue
        # Reconcile an uncertain prior write before attempting an append.
        existing = lark("+fetch", "--doc", DOC, "--scope", "keyword", "--keyword", key)
        content = existing["data"]["document"]["content"]
        if key not in content or row["created_at"] not in content:
            text = (f"{key} | {'有效' if row['valid'] else '无效'} | "
                    f"端到端 {row['output_tokens_per_second']:.3f} tokens/s | "
                    f"decode {row['tps_excluding_first_token']} tokens/s | "
                    f"成功 {row['successful_requests']}/{row['total_requests']} | "
                    f"实际输出 {row['completion_tokens_min']}–{row['completion_tokens_max']} tokens | "
                    f"总输出 {row['total_completion_tokens']} | wall {row['wall_time_seconds']:.3f}s | {row['created_at']}")
            args.spool.write_text("<p>" + escape(text) + "</p>")
            result = lark("+update", "--doc", DOC, "--command", "append", "--content", "@" + str(args.spool))
            data = result["data"]
            if data.get("result") != "success" or data.get("warnings"):
                raise RuntimeError(str(result))
            existing = lark("+fetch", "--doc", DOC, "--scope", "keyword", "--keyword", key)
            verified = existing["data"]["document"]["content"]
            if key not in verified or row["created_at"] not in verified:
                raise RuntimeError("Write verification failed: " + key)
        state[version_key] = row
        args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        print("SYNCED " + key, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, default=Path(__file__).resolve().parent / "sync_state.json")
    p.add_argument("--spool", type=Path, default=Path(__file__).resolve().parent / "sync_spool.xml")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    while True:
        try:
            sync(args)
        except Exception as exc:
            print("SYNC_ERROR " + str(exc), flush=True)
            if args.once:
                raise
        if args.once:
            break
        time.sleep(15)
