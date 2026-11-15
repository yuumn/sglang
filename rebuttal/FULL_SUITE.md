# Full-dataset benchmark requested in h20/tps.md

This is separate from the N32 tuning matrix `formal_flashinfer`. Finish and
publish the tuning comparison before starting this suite. No configuration in
that sampled matrix is eligible to skip the full-dataset stage. For a uniform
protocol, the full suite measures AIME25 again as well.

There are 12 configurations: Qwen3-4B and Qwen3-8B, each with `dflash`,
`dflash-3L`, `dspark`, `dspark_latentspec`, `eagle3`, and `latentspec`.
Each configuration has 120 measured cells: six complete datasets, concurrency
1/2/4/8/16, fixed output 512/1024/2048/4096. Dataset counts are 500, 30, 164,
256, 80, and 500 respectively. Total requested output is 705,024,000 tokens,
plus excluded warmup and preflight traffic. This is a long running benchmark.

On H20, after the current benchmark has released the GPU:

```bash
cd /root/workspace/sglang
# Verify the production fused initializer at the 8B hidden dimension too.
PYTHONPATH=python .venv/bin/python rebuttal/probe_init_fusion.py --production \
  --hidden-size 4096 --output rebuttal/output/init-fusion-8b-probe.json

# Must use exactly the same environment for preflight and final execution.
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_LATENTSPEC_FUSED_CACHED_INIT=1
.venv/bin/python -u rebuttal/full_suite.py --preflight
.venv/bin/python -u rebuttal/full_suite.py
```

For a detached run, use `nohup` with stdin redirected from `/dev/null` and a
log file in `rebuttal/output/`. Repeat the same final command to resume after
an interruption. Only a configuration with all 120 valid full-data cells and
its completion marker is skipped. A partial cell is always rerun completely;
the client never combines timing from separate attempts.

The runner holds an exclusive lock, checks the server port before staging any
weights, owns and stops its server process group, and retries each failed
configuration twice before moving on to the others. Any errors remain in
`errors.jsonl` and make the suite outcome incomplete. They must be investigated;
an incomplete outcome is never labelled successful.

The suite freezes a dataset snapshot, script hashes, framework patch, installed
packages, GPU identity, sampling protocol, and selected environment flags.
Resume rejects changes to these inputs. Source checkpoints are copied into
marked, disposable shared-memory directories and checked with SHA256 while
copying and again when reading back. Only the runner's marked model cache may
be replaced. Original checkpoints are preserved. Logs are gzip-compressed
after each configuration, including retries, to conserve the root filesystem.

`output/full_v1/measurements/<family>/<variant>/` contains the configuration,
model identities, metadata, per-request timings, cell summaries and compressed
server logs. `request-progress.json` is refreshed every 30 seconds during a
long cell. Its partial counters are telemetry, not performance results.
`export_full.py` emits a read-only compact JSON view for monitoring/publication.

On the local machine with authenticated `lark-cli`, run:

```bash
python3 sglang/rebuttal/publish_full.py
```

This publishes each complete 30-cell table to the second requested wiki:
`BtS7wZJ8ciF7lRkncqTc6XvynQA` (document `IPecd0Z1Yo8ZH4xzoN7cLCY0nRd`).
There will be exactly 48 result tables, one per target/draft/output length,
with six dataset rows and five concurrency columns. Every write is fetched
back and every displayed value is checked. A content hash prevents duplicate
tables after an uncertain write. The first document's raw-result synchronizer
must be stopped after the tuning stage is fully acknowledged, before this
full suite begins.

Local publication state, latest status, and a CSV are under
`h20/output/full_v1/`. `full-publication-complete.json` is created only when
1440 full-data cells, 48 verified tables, and a successful remote outcome all
exist. Launching a process does not mean that these measurements are complete.
