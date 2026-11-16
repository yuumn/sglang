#!/usr/bin/env python3
"""Fill the user's existing comparison-table layout with verified full-data TPS."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET

from full_common import DATASETS, COUNTS, CONCURRENCIES, LENGTHS, atomic_json
from sync_feishu import lark

DOC = "LKBcdEyM9oerZwxZIjdcphZonHG"
ALIASES = {"dflash": "dflash-5L", "dflash-3L": "dflash-3L", "dflash-4L":"dflash-4L", "eagle3": "eagle3",
           "dspark": "dspark", "dspark_latentspec": "dspark_latentspec", "latentspec": "latentspec"}
DATASET_LABELS = {"MATH-500": "math500", "AIME25": "aime25", "HumanEval": "humaneval",
                  "MBPP": "mbpp", "MT-Bench": "mt-bench", "Alpaca": "alpaca"}


def parse(content):
    return ET.fromstring("<root>" + (content or "") + "</root>")


def fetch():
    result = lark("+fetch", "--doc", DOC, "--detail", "full")
    return parse(result["data"]["document"]["content"])


def update(command, content, block_id=None):
    args = ["+update", "--doc", DOC, "--command", command, "--content", content]
    if block_id:
        args += ["--block-id", block_id]
    done = lark(*args)
    if done["data"].get("result") != "success" or done["data"].get("warnings"):
        raise RuntimeError(str(done))


def scan(root):
    family = length = None
    headings, families, tables = {}, {}, {}
    for node in root:
        text = "".join(node.itertext()).strip()
        if node.tag == "h1":
            family = "qwen3-4b" if text == "Qwen3-4B" else "qwen3-8b" if text == "Qwen3-8B" else None
            length = None
            if family:
                families[family] = node.attrib["id"]
        elif node.tag == "h2" and family:
            match = re.search(r"输出长度[：:]\s*(\d+)", text)
            if match:
                length = int(match.group(1))
                headings[family, length] = node.attrib["id"]
        elif node.tag == "table" and family:
            # The supplied 8B template initially omitted its 512 heading.
            tables.setdefault((family, length or 512), []).append(node)
    return families, headings, tables


def cells(table):
    return [[cell for cell in row if cell.tag in ("td", "th")] for row in table.iter("tr")]


def coordinate(table, variant, dataset, concurrency):
    rows = cells(table)
    pair = [DATASET_LABELS["".join(c.itertext()).strip()] for c in rows[0][1:]]
    if dataset not in pair:
        return None
    label = ALIASES[variant].casefold()
    ri = next(i for i, row in enumerate(rows[2:], 2) if "".join(row[0].itertext()).strip().casefold() == label)
    ci = 1 + pair.index(dataset) * 5 + CONCURRENCIES.index(concurrency)
    return ri, ci


def prepare_layout(args):
    root = fetch()
    _, _, tables = scan(root)
    templates = tables["qwen3-4b", 512]
    assert len(templates) == 3
    copies = []
    for table in templates:
        clone = deepcopy(table)
        for node in clone.iter():
            node.attrib.pop("id", None)
        # Clone the format, leaving measured-data cells blank.
        for row in cells(clone)[2:]:
            for cell in row[1:]:
                for p in cell:
                    p.clear()
        copies.append(ET.tostring(clone, encoding="unicode"))
    xml = "".join(copies)
    for family in ("qwen3-4b", "qwen3-8b"):
        for i, length in enumerate(args.lengths):
            families, headings, tables = scan(fetch())
            key = family, length
            if key not in headings:
                anchor = families[family] if i == 0 else tables[family, args.lengths[i - 1]][-1].attrib["id"]
                update("block_insert_after", f"<h2>输出长度：{length}</h2>", anchor)
                _, headings, tables = scan(fetch())
            if not tables.get(key):
                update("block_insert_after", xml, headings[key])
                _, _, tables = scan(fetch())
            if len(tables[key]) != 3:
                raise RuntimeError(f"Unexpected table layout for {key}")
    root = fetch()
    if args.priority:
        marker = "PRIORITY_ORDER_v2"
        if marker not in "".join(root.itertext()):
            update("append", "<h1>测试顺序与配置记录</h1><p>PRIORITY_ORDER_v2：先执行4B的dspark、dspark_latentspec在Math500/AIME25的全量测试，再执行8B同组测试；"
                   "并发均为1/2/4/8/16，输出512/2048/4096。随后依次执行4B、8B的latentspec与dflash-5L在HumanEval/MBPP/MT-Bench/Alpaca的输出512全量测试。"
                   "四阶段共200项；其余前文要求的配置排在四阶段之后。每组配置写入文档并回读确认后再进入下一组。"
                   "所有新增测试固定Decode CUDA Graph最大batch=256、max_running_requests=256、FlashInfer工作区1GiB，使用云端最新代码；实际提交记录在原始manifest。"
                   "已完成的4B dflash-5L结果保留原Graph32来源，共69项符合本次512/2048/4096范围；其中20项属于四个优先阶段。"
                   "旧1024结果另存历史归档，不加入本次测量范围。单位为tokens/s，实际总输出tokens除以整组请求墙钟时间，预热不计入。"
                   "dflash对应实际5层检查点；4L暂无本次实测，保持空白。空格表示未完成全量测试，不填入调优抽样结果。</p>")
            assert marker in "".join(fetch().itertext())
    elif "CG256_CONFIG_SWITCH_v1" not in "".join(root.itertext()):
        update("append", "<h1>测试口径与配置记录</h1><p>CG256_CONFIG_SWITCH_v1：数值单位为tokens/s，"
               "使用六个本地数据文件全部行，统计实际总输出tokens除以整组请求墙钟时间，预热不计入。"
               "dflash检查点实际为5层，对应dflash-5L行；dflash-4L暂无本次实测结果，保留为空。"
               "旧配置Decode CUDA Graph最大batch=32、max_running_requests=32的99格结果保留："
               "4B dflash-5L的512/1024/2048全部90格，以及4096的并发1六个数据集和并发2的Math500、AIME25、HumanEval九格。"
               "之后使用Decode CUDA Graph最大batch=256、max_running_requests=256、FlashInfer工作区1GiB；"
               "实际压测并发仍为1/2/4/8/16。旧值保留原始来源，不标作256配置实测。"
               "空白单元格表示尚未完成全量测量；前期32条调优抽样不填入本表。</p>")
        assert "CG256_CONFIG_SWITCH_v1" in "".join(fetch().itertext())
    print("LAYOUT_READY comparison tables, original column widths and merged headers preserved", flush=True)


def fetch_table(table_id):
    result = lark("+fetch", "--doc", DOC, "--scope", "range", "--start-block-id", table_id,
                  "--end-block-id", table_id, "--detail", "full")
    return next(parse(result["data"]["document"]["content"]).iter("table"))


def sync(args):
    command = "/root/workspace/sglang/.venv/bin/python /root/workspace/sglang/rebuttal/" + args.export_script + " --output " + shlex.quote("/root/workspace/sglang/rebuttal/output/" + args.phase)
    if args.payload_file:
        payload = json.loads(args.payload_file.read_text())
    else:
        done = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "h20", command], check=True,
                              capture_output=True, text=True, timeout=90)
        payload = json.loads(done.stdout)
    atomic_json(args.output / "full-status.json", payload)
    state_path = args.output / "template-publication-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"doc_id": DOC, "cells": {}}
    assert state["doc_id"] == DOC
    _, _, tables = scan(fetch())
    grouped = {}
    for row in payload["rows"]:
        key = "/".join(map(str, (row["family"], row["variant"], row["dataset"], row["concurrency"], row["output_length"])))
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if state["cells"].get(key, {}).get("sha256") == digest:
            continue
        candidates = tables.get((row["family"], row["output_length"]), [])
        target = next(t for t in candidates if coordinate(t, row["variant"], row["dataset"], row["concurrency"]) is not None)
        grouped.setdefault(target.attrib["id"], []).append((key, digest, row))
    for table_id, pending in grouped.items():
        table = fetch_table(table_id)
        for key, digest, row in pending:
            ri, ci = coordinate(table, row["variant"], row["dataset"], row["concurrency"])
            cell = cells(table)[ri][ci]
            old = "".join(cell.itertext()).strip()
            value = f"{row['output_tokens_per_second']:.3f}"
            previous = state["cells"].get(key, {}).get("displayed")
            if old and old != value and old != previous:
                raise RuntimeError(f"Existing user value conflicts at {key}: {old}")
            if old != value:
                paragraph = next(cell.iter("p"))
                update("block_replace", "<p>" + escape(value) + "</p>", paragraph.attrib["id"])
                table = fetch_table(table_id)
                check_ri, check_ci = coordinate(table, row["variant"], row["dataset"], row["concurrency"])
                assert "".join(cells(table)[check_ri][check_ci].itertext()).strip() == value
            state["cells"][key] = {"sha256": digest, "displayed": value,
                                   "decode_cuda_graph_max_bs": row.get("decode_cuda_graph_max_bs"),
                                   "source_case_id": row["case_id"]}
            atomic_json(state_path, state)
            print("FILLED " + key + " = " + value, flush=True)
    statistics_ready=None
    if args.request_statistics:
        from publish_request_statistics import publish
        statistics_ready=publish(args,payload,state,fetch,fetch_table,update,DOC)
    if args.priority and not args.payload_file:
        acks = []
        for unit in payload.get("units", []):
            if unit["complete"] and (statistics_ready is None or unit["unit_id"] in statistics_ready) and all(state["cells"].get(k, {}).get("sha256") == v
                                        for k,v in unit["versions"].items()):
                acks.append({"unit_id":unit["unit_id"],"sha256":unit["sha256"],"doc_id":DOC,
                             "verified_at":datetime.now(timezone.utc).isoformat()})
        if acks:
            script = "import json,sys;from pathlib import Path;sys.path.insert(0,'/root/workspace/sglang/rebuttal');from full_common import atomic_json;out=Path('/root/workspace/sglang/rebuttal/output/" + args.phase + "/publication-acks');out.mkdir(exist_ok=True);[(atomic_json(out/(a['unit_id']+'.json'),a)) for a in json.load(sys.stdin)]"
            subprocess.run(["ssh","-o","ConnectTimeout=15","h20","/root/workspace/sglang/.venv/bin/python -c " + shlex.quote(script)],
                           input=json.dumps(acks),text=True,capture_output=True,check=True,timeout=60)
    total = payload.get("total_cells",1440)
    acknowledged = sum(state["cells"].get("/".join(map(str,(r["family"],r["variant"],r["dataset"],r["concurrency"],r["output_length"]))),{}).get("sha256") == hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest() for r in payload["rows"])
    complete = len(payload["rows"]) == total and acknowledged == total and (payload.get("outcome") or {}).get("complete")
    print(f"STATUS measured={len(payload['rows'])}/{total} filled={acknowledged}/{total} complete={bool(complete)}", flush=True)
    if complete:
        assert sum(r["total_completion_tokens"] for r in payload["rows"]) == payload.get("expected_completion_tokens",705_024_000)
        marker = "REQUEST_STATISTICS_COMPLETE_v3" if args.request_statistics else "PRIORITY_FULL_COMPLETE_v2" if args.priority else "TEMPLATE_FULL_COMPLETE_v1"
        if args.phase=="dflash4l_v1":
            marker="DFLASH_4L_COMPLETE_v1"
        if marker not in "".join(fetch().itertext()):
            message=f"：本轮全部{total}格全量结果已测完并逐格回读核验。"
            message+=f"全部{total}项均有真实结束原因和实际输出长度，{len(payload.get('units',[]))}种目标/草稿配置的逐请求CSV已上传；Graph256。" if args.request_statistics else "保留的旧结果按原Graph32来源记录，新增测量使用Graph256。"
            update("append", "<p>" + marker + message + "</p>")
            assert marker in "".join(fetch().itertext())
        atomic_json(args.output / "full-publication-complete.json", {"cells": total, "doc_id": DOC,
                    "verified_at": datetime.now(timezone.utc).isoformat()})
    return complete


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", default="full_cg256")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "h20/output/full_cg256")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--payload-file", type=Path, help="Publish a saved, verified checkpoint while the benchmark host is unavailable")
    p.add_argument("--doc", default=DOC)
    p.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    p.add_argument("--export-script", choices=["export_full.py","export_priority.py"],default="export_full.py")
    p.add_argument("--priority", action="store_true")
    p.add_argument("--request-statistics", action="store_true")
    args = p.parse_args()
    DOC = args.doc
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.phase):
        p.error("Invalid phase")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "template-publication.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.prepare:
        prepare_layout(args)
    while True:
        try:
            done = sync(args)
        except Exception as exc:
            print("PUBLISH_ERROR " + str(exc), flush=True)
            if args.once:
                raise
            done = False
        if args.once or done:
            break
        time.sleep(30)
