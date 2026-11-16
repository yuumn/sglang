# DFlash four-layer full-dataset benchmark

Task: `h20/tps-4l.md`. The baseline branch `origin/rebuttal` was independently
verified through `git ls-remote` as commit
`fd0b4858f4211eab171315d3fbc183c79ccceb2e`. DFlash4L support is committed
on top of that baseline before execution; the exact run commit is saved in
the phase manifest.
The direct H20-to-GitHub fetch timed out; the successful local lookup verified
the branch head without changing the source version.

Target: `/mydata/Qwen/Qwen3-4B`.
Draft: `/mydata/models/qwen3-4b/dflash-4L/step_2616`.
Its config declares `Qwen3DSparkModel`, hidden size2560 and4 draft layers.
The launcher selects DFLASH,8 draft tokens,prediction-hidden-start0,
BF16,target FlashInfer,draft Triton,TP1,Graph256,max-running-requests256,
1GiB FlashInfer workspace and context16384. The runner checks actual graph
capture and the draft layer count before publishing measurements.

The explicit plan contains30 cells: six full datasets, concurrency1/2/4/8/16,
fixed output512. This is7650 requests and3,916,800 measured output tokens.
No previous five-layer DFlash or DSpark measurements are inherited.
Warmups are excluded, partial cells rerun completely and real terminal reasons
are recorded for token-length statistics and truncation rates.

The phase is `rebuttal/output/dflash4l_v1`. On H20:

```bash
bash rebuttal/run_dflash4l.sh
```

The entry script creates the explicit plan and an empty inheritance file when
needed, so a fresh checkout can run without copying ignored output files.

On the local machine:

```bash
python3 h20/run_dflash4l.py --watch
bash h20/poll_dflash4l.sh
```

The watcher waits for NVIDIA devices and usable CUDA, checks the prepared
commit and file hashes, leaves other servers untouched, then starts the
detached runner and local publisher. A changed workspace stops it for
inspection. Current state is `h20/output/dflash4l_v1/start-state.json`.
Model weights are copied and SHA256-checked into owned shared-memory caches;
missing caches after a container restart are rebuilt.

Results go into the existing `dflash-4L` row of the three 4B/output512 tables at
<https://d1rw2z9pe2g.feishu.cn/wiki/OvcPwxg3UiROV8kcYCocPAkinyc>.
The publisher fetch-verifies every value, adds output-length/truncation
statistics and uploads the complete7650-request CSV before acknowledging
completion. GPU unavailability is never recorded as a benchmark result.

## Three-layer continuation

After the four-layer phase is complete and its publication is verified, run
`bash rebuttal/run_dflash3l.sh` for the same30 full-data cells using
`/mydata/models/qwen3-4b/dflash-3L/step_2616`. Its independent phase is
`dflash3l_v1`; no4L results are inherited. Publish with the same local CLI
arguments, changing the phase/output to `dflash3l_v1`. Values are written into
the existing `dflash-3L` row, with separate output statistics and request CSV.
