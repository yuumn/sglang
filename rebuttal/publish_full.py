#!/usr/bin/env python3
"""Run locally: publish 48 complete full-dataset tables, fetch-verify each write.

Incomplete tables stay in the local export and are not labelled final results.
An uncertain write is reconciled by fetching its unique heading and content hash.
"""
import argparse
import csv
from datetime import datetime, timezone
from html import escape
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

from full_common import DATASETS, COUNTS, VARIANTS, CONCURRENCIES, LENGTHS, atomic_json
from sync_feishu import lark

DOC = "IPecd0Z1Yo8ZH4xzoN7cLCY0nRd"
LABELS = dict(zip(DATASETS, ["MATH-500", "AIME25", "HumanEval", "MBPP", "MT-Bench", "Alpaca"]))


def fetch_table(key):
    found = lark("+fetch", "--doc", DOC, "--scope", "keyword", "--keyword", key, "--detail", "with-ids")
    content = found["data"]["document"]["content"]
    if not content or not content.strip():
        return None
    root = ET.fromstring(content)
    headings = [node for node in root.iter("h2") if key in "".join(node.itertext())]
    if not headings:
        return None
    if len(headings) != 1:
        raise RuntimeError("Duplicate table heading: " + key)
    section = lark("+fetch", "--doc", DOC, "--scope", "section",
                   "--start-block-id", headings[0].attrib["id"], "--detail", "with-ids")
    return section["data"]["document"]["content"]


def write_xml(args, xml):
    args.spool.write_text(xml)
    result = lark("+update", "--doc", DOC, "--command", "append", "--content", "@" + str(args.spool))
    if result["data"].get("result") != "success" or result["data"].get("warnings"):
        raise RuntimeError(str(result))


def verify_section(content, marker, values):
    root = ET.fromstring(content)
    if marker not in "".join(root.itertext()):
        raise RuntimeError("Missing table provenance marker")
    tables = list(root.iter("table"))
    if len(tables) != 1:
        raise RuntimeError(f"Expected one result table, got {len(tables)}")
    rows = [["".join(cell.itertext()).strip() for cell in row if cell.tag in ("td", "th")]
            for row in tables[0].iter("tr")]
    if rows != values:
        raise RuntimeError("Published table differs from measured values: " + repr(rows))


def sync(args):
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.spool.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    done = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "h20",
                           "/root/workspace/sglang/.venv/bin/python /root/workspace/sglang/rebuttal/export_full.py"],
                          check=True, capture_output=True, text=True, timeout=60)
    payload = json.loads(done.stdout)
    atomic_json(args.state.parent / "full-status.json", payload)
    rows = payload["rows"]
    if not rows:
        print("WAIT no complete full-dataset cells", flush=True)
        return False
    # Fetch the document before its first edit, preserving any user content.
    if not state.get("methods"):
        existing = lark("+fetch", "--doc", DOC, "--detail", "with-ids")
        content = existing["data"]["document"]["content"]
        marker = "FULL_MATRIX_METHODS_v1"
        if marker not in content:
            write_xml(args, "<h1>Qwen3 全量吞吐测试</h1><p>" + marker +
                      "：两种目标模型×六种草稿模型×四档固定输出×五档并发，共1440格、48张表。"
                      "每格使用该数据集全部行：MATH-500 500、AIME25 30、HumanEval 164、MBPP 256、MT-Bench 80、Alpaca 500。"
                      "表内单位为tokens/s，指标为实际总输出tokens除以整组请求墙钟时间，包含首token；"
                      "固定输出设置ignore_eos=true，每条请求核验长度，无失败请求才有效。"
                      "BF16、TP=1、H20-3e、target FlashInfer、draft Triton、草稿块8、静态验证、temperature=1、seed=980406、关闭thinking。"
                      "预热不计入测量，测量前清空prefix cache。预检和调优抽样均不算全量结果。"
                      "每张表30格完整后发布；尚未出现的表仍在运行。原始逐请求计时、配置指纹、数据SHA256和模型SHA256保存在"
                      "/root/workspace/sglang/rebuttal/output/full_v1/。单次测量，无误差区间。</p>")
            verified = lark("+fetch", "--doc", DOC, "--scope", "keyword", "--keyword", marker)
            if marker not in verified["data"]["document"]["content"]:
                raise RuntimeError("Methods verification failed")
        state["methods"] = True
        atomic_json(args.state, state)
    indexed = {(r["family"], r["variant"], r["output_length"], r["dataset"], r["concurrency"]): r for r in rows}
    for family in ["qwen3-4b", "qwen3-8b"]:
        for variant in VARIANTS:
            for length in LENGTHS:
                key = f"FULLTABLE {family}/{variant}/fixed{length}"
                selected = [indexed.get((family, variant, length, d, c)) for d in DATASETS for c in CONCURRENCIES]
                if any(r is None for r in selected):
                    continue
                digest = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
                marker = "FULL_TABLE_SHA256=" + digest
                if state.get(key) == digest:
                    continue
                values = [["数据集（全量条数）"] + [f"并发 {c}" for c in CONCURRENCIES]]
                for d in DATASETS:
                    values.append([f"{LABELS[d]} ({COUNTS[d]})"] + [
                        f"{indexed[family, variant, length, d, c]['output_tokens_per_second']:.3f}" for c in CONCURRENCIES])
                xml = "<h2>" + escape(key) + "</h2><p>" + marker + "</p><table><thead><tr>"
                xml += "".join("<th><p>" + escape(v) + "</p></th>" for v in values[0]) + "</tr></thead><tbody>"
                for row in values[1:]:
                    xml += "<tr>" + "".join("<td><p>" + escape(v) + "</p></td>" for v in row) + "</tr>"
                xml += "</tbody></table>"
                existing = fetch_table(key)
                if existing is None:
                    write_xml(args, xml)
                    existing = fetch_table(key)
                verify_section(existing, marker, values)
                state[key] = digest
                atomic_json(args.state, state)
                print("PUBLISHED " + key, flush=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["family", "variant", "dataset", "concurrency", "output_length", "total_requests",
              "total_completion_tokens", "wall_time_seconds", "output_tokens_per_second", "case_id", "created_at"]
    with args.csv.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tables = sum(key.startswith("FULLTABLE ") for key in state)
    complete = len(rows) == 1440 and tables == 48 and (payload.get("outcome") or {}).get("complete")
    print(f"STATUS {len(rows)}/1440 full cells; {tables}/48 published tables; complete={bool(complete)}", flush=True)
    if complete:
        marker = "FULL_MATRIX_COMPLETE_v1"
        total_tokens = sum(row["total_completion_tokens"] for row in rows)
        if total_tokens != 705_024_000:
            raise RuntimeError(f"Unexpected completed output token total: {total_tokens}")
        existing = lark("+fetch", "--doc", DOC, "--scope", "keyword", "--keyword", marker)
        content = existing["data"]["document"]["content"]
        if marker not in content:
            write_xml(args, "<h2>全量测试完成</h2><p>" + marker +
                      "：全部1440项全量测量已完成，48张结果表均已发布并回读核验。"
                      "两种目标模型与六种草稿模型均覆盖六个本地数据集的全部行、五档并发和四档固定输出。"
                      "正式测量实际总输出为705024000 tokens，所有请求成功且逐条满足固定输出长度。"
                      "预热、预检与前期调优抽样未计入这些结果。原始逐请求计时、模型和数据SHA256、"
                      "冻结配置及压缩服务日志保存在H20的rebuttal/output/full_v1/。</p>")
            existing = lark("+fetch", "--doc", DOC, "--scope", "keyword", "--keyword", marker)
            content = existing["data"]["document"]["content"]
        if not all(value in content for value in [marker, "1440", "48", "705024000"]):
            raise RuntimeError("Final completion notice verification failed")
        state["completion_notice_verified"] = True
        atomic_json(args.state, state)
        atomic_json(args.state.parent / "full-publication-complete.json", {
            "cells": len(rows), "tables": tables, "doc_id": DOC,
            "verified_at": datetime.now(timezone.utc).isoformat()})
    return complete


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    output = Path(__file__).resolve().parents[2] / "h20/output/full_v1"
    p.add_argument("--state", type=Path, default=output / "publication-state.json")
    p.add_argument("--spool", type=Path, default=output / "publication-spool.xml")
    p.add_argument("--csv", type=Path, default=output / "full-results.csv")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock = args.state.with_suffix(".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            complete = sync(args)
        except Exception as exc:
            print("PUBLISH_ERROR " + str(exc), flush=True)
            if args.once:
                raise
            complete = False
        if args.once or complete:
            break
        time.sleep(30)
