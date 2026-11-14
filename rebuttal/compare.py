#!/usr/bin/env python3
"""Compare matched valid cells and emit auditable CSV/JSON."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from html import escape


def read(path):
    rows = {}
    for file in Path(path).glob("**/results.jsonl"):
        for line in file.read_text().splitlines():
            r = json.loads(line)
            if not r["valid"]:
                continue
            key = (r["dataset"], r["concurrency"], r["output_length"])
            if key in rows:
                raise ValueError(f"Multiple valid results for {key} under {path}")
            rows[key] = r
    return rows


def compare(args):
    baseline, candidate = read(args.baseline), read(args.candidate)
    rows = []
    for key in sorted(baseline.keys() & candidate.keys()):
        b, c = baseline[key], candidate[key]
        if b["total_requests"] != c["total_requests"] or b["sampling"] != c["sampling"]:
            raise ValueError(f"Unmatched benchmark settings: {key}")
        rows.append(dict(dataset=key[0], concurrency=key[1], output_length=key[2],
                         requests=b["total_requests"], baseline_tps=b["output_tokens_per_second"],
                         candidate_tps=c["output_tokens_per_second"],
                         ratio=c["output_tokens_per_second"] / b["output_tokens_per_second"]))
    if not rows:
        raise ValueError("No matched valid cells")
    ratios = [r["ratio"] for r in rows]
    summary = dict(matched=len(rows), baseline_cells=len(baseline), candidate_cells=len(candidate),
                   missing_candidate=[list(k) for k in sorted(baseline.keys() - candidate.keys())],
                   missing_baseline=[list(k) for k in sorted(candidate.keys() - baseline.keys())],
                   geometric_mean_ratio=math.exp(statistics.fmean(map(math.log, ratios))),
                   min_ratio=min(ratios), max_ratio=max(ratios),
                   wins=sum(r > 1 for r in ratios), losses=sum(r <= 1 for r in ratios))
    for name, path in [("baseline", args.baseline), ("candidate", args.candidate)]:
        config_path = path / "benchmark-config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            summary[name + "_expected_cells"] = len(config["datasets"]) * len(config["concurrencies"]) * len(config["lengths"])
    summary["complete"] = (not summary["missing_candidate"] and not summary["missing_baseline"]
                           and summary["matched"] == summary.get("baseline_expected_cells", -1)
                           and summary["matched"] == summary.get("candidate_expected_cells", -1))
    if args.require_complete and not summary["complete"]:
        raise ValueError("Comparison matrix is incomplete: " + json.dumps(summary))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "comparison.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.output / "comparison.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    xml = ["<h2>" + escape(args.title) + "</h2>",
           f"<p>匹配单元格 {len(rows)}；几何平均吞吐比 {summary['geometric_mean_ratio']:.4f}；胜/负 {summary['wins']}/{summary['losses']}；最小/最大比值 {min(ratios):.4f}/{max(ratios):.4f}。矩阵{'完整' if summary['complete'] else '尚未完整'}。比值为candidate/baseline，使用端到端总输出tokens/wall seconds。</p>"]
    for length in sorted({r["output_length"] for r in rows}):
        subset = [r for r in rows if r["output_length"] == length]
        concurrencies = sorted({r["concurrency"] for r in subset})
        xml.append("<h3>" + ("自然停止（上限512）" if length == 0 else "固定输出 " + str(length)) + "</h3>")
        xml.append("<p>单元格：baseline TPS / candidate TPS（变化百分比）</p><table><thead><tr><th><p>数据集</p></th>" + "".join(f"<th><p>并发{c}</p></th>" for c in concurrencies) + "</tr></thead><tbody>")
        for dataset in sorted({r["dataset"] for r in subset}):
            xml.append("<tr><td><p>" + escape(dataset) + "</p></td>")
            for concurrency in concurrencies:
                r = next((r for r in subset if r["dataset"] == dataset and r["concurrency"] == concurrency), None)
                text = "缺失" if r is None else f"{r['baseline_tps']:.2f} / {r['candidate_tps']:.2f} ({100*(r['ratio']-1):+.2f}%)"
                xml.append("<td><p>" + escape(text) + "</p></td>")
            xml.append("</tr>")
        xml.append("</tbody></table>")
    (args.output / "comparison.xml").write_text("".join(xml))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default="DSPARK 与 DSPARK_LATENTSPEC 吞吐对比")
    p.add_argument("--require-complete", action="store_true")
    compare(p.parse_args())
