#!/usr/bin/env python3
"""Summarize a complete tuning matrix, with requested fixed-length cells separate."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


def summary(rows):
    ratios = [row["ratio"] for row in rows]
    return {"pairs": len(rows), "wins": sum(r > 1 for r in ratios),
            "geometric_mean_ratio": math.exp(statistics.fmean(map(math.log, ratios))),
            "min_ratio": min(ratios), "max_ratio": max(ratios)}


def main(args):
    data = json.loads(args.comparison.read_text())
    if not data["summary"]["complete"] or len(data["rows"]) != 150:
        raise ValueError("Need the complete 150-pair tuning comparison")
    fixed = [row for row in data["rows"] if row["output_length"] > 0]
    if len(fixed) != 120:
        raise ValueError("Need exactly 120 requested fixed-length scenarios")
    result = {"overall": summary(data["rows"]), "fixed": summary(fixed),
              "by_output_length": {str(n): summary([r for r in data["rows"] if r["output_length"] == n])
                                   for n in [0, 512, 1024, 2048, 4096]},
              "by_concurrency_fixed": {str(c): summary([r for r in fixed if r["concurrency"] == c])
                                       for c in [1, 2, 4, 8, 16]},
              "fixed_majority_faster": sum(r["ratio"] > 1 for r in fixed) > len(fixed) / 2,
              "created_at": datetime.now(timezone.utc).isoformat()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tuning-summary.json").write_text(json.dumps(result, indent=2))
    all_, f = result["overall"], result["fixed"]
    xml = ("<h2>最终调优验证结论</h2><p>FINAL_TUNING_ANALYSIS_v1："
           "本阶段每格取相同的前32条提示词，AIME25仅30条，属于调优验证抽样；不代表后续全量数据测试。"
           f"用户要求的固定长度场景共{f['pairs']}对，DSPARK_LATENTSPEC胜出{f['wins']}对，"
           f"几何平均吞吐比{f['geometric_mean_ratio']:.4f}，"
           f"最小/最大比值{f['min_ratio']:.4f}/{f['max_ratio']:.4f}。"
           f"包含自然停止补充测试的全部{all_['pairs']}对中胜出{all_['wins']}对，"
           f"几何平均吞吐比{all_['geometric_mean_ratio']:.4f}。"
           "比值均为DSPARK_LATENTSPEC / DSPARK，指标为实际总输出tokens / 整组请求墙钟时间。"
           "两者共用FlashInfer目标注意力、Triton草稿注意力、BF16、TP1、块长8和静态完整验证。"
           "LatentSpec缓存初始化已融合为单个Triton kernel，生产路径逐位一致性和CUDA Graph重放检查通过；"
           "该局部kernel加速不等同于端到端吞吐提升倍数。每格测量一次，无置信区间。"
           "随后按新任务对4B/8B与六种草稿进行全部数据行的独立测试。</p>")
    xml += "<table><thead><tr><th><p>输出策略</p></th><th><p>胜出/配对</p></th><th><p>几何平均吞吐比</p></th></tr></thead><tbody>"
    for length, s in result["by_output_length"].items():
        label = "自然停止（上限512）" if length == "0" else "固定 " + length
        xml += f"<tr><td><p>{label}</p></td><td><p>{s['wins']}/{s['pairs']}</p></td><td><p>{s['geometric_mean_ratio']:.4f}</p></td></tr>"
    xml += "</tbody></table>"
    (args.output / "tuning-summary.xml").write_text(xml)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
