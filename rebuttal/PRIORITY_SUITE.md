# Updated execution order from h20/tps.md

Remote branch `rebuttal` was fetched and fast-forwarded to
`6c6cf45c6dc78605c043086aa240ae992dd754aa`. The previous six modified files
were identical to the incoming branch versions; their patch and stash were
preserved under `output/priority_v2/before-cloud-update/`.

The first eight units follow the four newly requested stages:

| Stage | Target | Drafts, in order | Complete datasets | Fixed outputs |
|---|---|---|---|---|
| 1 | Qwen3-4B | dspark, dspark_latentspec | Math500, AIME25 | 512, 2048, 4096 |
| 2 | Qwen3-8B | dspark, dspark_latentspec | Math500, AIME25 | 512, 2048, 4096 |
| 3 | Qwen3-4B | latentspec, dflash-5L | HumanEval, MBPP, MT-Bench, Alpaca | 512 |
| 4 | Qwen3-8B | latentspec, dflash-5L | HumanEval, MBPP, MT-Bench, Alpaca | 512 |

All units use concurrency 1/2/4/8/16 and every row of each selected dataset.
The first four stages contain 200 cells and 80,793,600 output tokens, including
20 preserved old DFlash cells. The preceding full-matrix requirement is retained
after these four stages: 1080 unique cells total, excluding output1024.
`execution-control.json` may set `stop_after_stage` to4 if the user narrows the
scope to the explicit four stages. This changes scheduling, not measured inputs.

All99 old records remain archived unchanged.69 match the new output lengths
and are explicitly inherited with their original Graph32 provenance. New
measurements use Graph256, max-running-requests256 and FlashInfer workspace1GiB.
Model checkpoint bytes, corpus hashes, packages, source, sampling and script
hashes are frozen in `output/priority_v2/suite-manifest.json`.

Start or resume on H20:

```bash
env SGLANG_RAGGED_VERIFY_MODE=static SGLANG_LATENTSPEC_FUSED_CACHED_INIT=1 \
  SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824 \
  .venv/bin/python -u rebuttal/priority_suite.py
```

The runner owns its server and client process groups, captures actual batch256,
reruns partial cells completely and waits for the local publisher's verified
acknowledgement after each unit. Two failed attempts stop execution at that
unit, preserving the requested order. Model copies use marked shared-memory
directories and SHA256 verification. Neither original checkpoints nor earlier
benchmark output is overwritten.

The current destination is
<https://d1rw2z9pe2g.feishu.cn/wiki/OvcPwxg3UiROV8kcYCocPAkinyc>,
document `Z9EWd3sAzouWR5xNZhcccMrHncd`. Its original wide-table format is
preserved; missing tables are cloned from its template. Start publishing locally:

```bash
python3 sglang/rebuttal/publish_template.py --phase priority_v2 \
  --output h20/output/priority_v2 --doc Z9EWd3sAzouWR5xNZhcccMrHncd \
  --lengths 512 2048 4096 --priority --export-script export_priority.py
bash h20/poll_full.sh
```

The preparation pass uses the same arguments plus `--prepare --once`.
Every displayed value is fetched back; an acknowledgement includes the exact
result versions. A successful process launch is not a completed benchmark.
