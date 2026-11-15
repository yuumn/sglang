#!/usr/bin/env python3
"""Plot complete paired throughput ratios as PNG/PDF/SVG research artifacts."""
import argparse
import json
import math
from pathlib import Path
import statistics


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(args.comparison.read_text())
    if not data["summary"]["complete"]:
        raise ValueError("Only a complete paired matrix may be plotted")
    rows = data["rows"]
    datasets = sorted({r["dataset"] for r in rows})
    labels = {"math500": "MATH-500", "aime25": "AIME25", "humaneval": "HumanEval",
              "mbpp": "MBPP", "mt-bench": "MT-Bench", "alpaca": "Alpaca"}
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    markers = ["o", "s", "^", "D", "v", "P"]
    lengths = sorted({r["output_length"] for r in rows})
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    handles = []
    for ax, length in zip(axes.flat, lengths):
        ax.axhline(1.0, color="0.45", linestyle="--", linewidth=1, zorder=0)
        for i, dataset in enumerate(datasets):
            selected = sorted([r for r in rows if r["output_length"] == length and r["dataset"] == dataset],
                              key=lambda r: r["concurrency"])
            line, = ax.plot([r["concurrency"] for r in selected], [r["ratio"] for r in selected],
                            label=labels.get(dataset, dataset), color=palette[i % len(palette)],
                            marker=markers[i % len(markers)], linewidth=1.5, markersize=4)
            if length == lengths[0]:
                handles.append(line)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("DSparkLatentSpec / DSpark throughput")
        ax.set_title("Natural EOS (cap 512)" if length == 0 else f"Fixed {length} output tokens")
        ax.grid(axis="y", alpha=0.2)
    last = axes.flat[-1]
    for unused in list(axes.flat)[len(lengths):-1]:
        unused.axis("off")
    last.axis("off")
    last.legend(handles=handles, loc="upper left", frameon=False, ncol=2)
    ratios = [r["ratio"] for r in rows]
    geo = math.exp(statistics.fmean(map(math.log, ratios)))
    counts = sorted({r["requests"] for r in rows})
    sample_label = " / ".join(map(str, counts))
    last.text(0.02, 0.57,
              f"Paired cells: {len(rows)}\nGeometric mean ratio: {geo:.4f}\n"
              f"Faster cells: {sum(r > 1 for r in ratios)} / {len(rows)}\n"
              f"Requests per cell: {sample_label} (matched prefixes)\n"
              "Metric: completion tokens / entire group wall time\n"
              "One measurement per cell; no confidence intervals.",
              transform=last.transAxes, fontsize=9, va="top", linespacing=1.6)
    fig.suptitle("Qwen3-4B on NVIDIA H20-3e", fontsize=14)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(args.output / ("throughput_ratio." + suffix), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
