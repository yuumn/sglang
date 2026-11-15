# H20 Qwen3-4B throughput comparison

The completed `formal_flashinfer` tuning validation measured 150 matched pairs
with the same first 32 prompts per dataset (AIME25: all 30 prompts). For the
120 requested fixed-length scenarios, DSparkLatentSpec was faster in 99/120,
with a geometric mean throughput ratio of 1.0616. It was faster in a majority
of the scenarios at every concurrency. Including the natural-EOS supplement,
it was faster in 124/150, with geometric mean ratio 1.0562. Each cell is one
measurement; these are sampled tuning results, not full-dataset measurements.
The comparison CSV/JSON, summary, plots, source snapshot and verified document
readbacks are in `output/formal_flashinfer/`. All 494 historical/tuning case
records and the final comparison tables and figure were published and fetched
back from the first requested Feishu document.

The subsequent two-target/six-draft full-dataset task has a separate launcher,
output tree and publisher; see [FULL_SUITE.md](FULL_SUITE.md). Its final scope
is 1440 all-row measurements and 48 tables. Sampled tuning configurations are
not skipped as fully completed configurations.

Run on `h20` from `/root/workspace/sglang`, using `.venv`.

```bash
# Optional first-run cache for the S3-mounted model files; verifies SHA256.
.venv/bin/python rebuttal/stage_models.py
rebuttal/launch_server.sh dspark
rebuttal/launch_server.sh dspark_latentspec

# Owns and stops each server process group; do not run another GPU workload.
.venv/bin/python -u rebuttal/run_matrix.py --phase formal_flashinfer --requests 32

# Resume: repeat the same command. Valid completed cells are skipped.
# --requests 0 uses all dataset prompts; --lengths 0 means natural EOS, cap 512.
.venv/bin/python rebuttal/status.py --phase formal_flashinfer
.venv/bin/python rebuttal/compare.py \
  --baseline rebuttal/output/formal_flashinfer/dspark \
  --candidate rebuttal/output/formal_flashinfer/dspark_latentspec \
  --output rebuttal/output/formal_flashinfer/comparison
```

The default matrix is six datasets × five concurrencies × five output policies
(natural EOS with cap 512, fixed 512/1024/2048/4096) × two variants = 300 cells.
Prompts are the same first 32 entries (AIME25 has 30). Sampling follows the
reference script: temperature 1, top_p 1, seed 980406, thinking disabled.
Both models use gamma 7 and static full verification. Target attention uses
FlashInfer and draft attention uses Triton, with BF16 weights/KV, TP=1 and the
same CUDA graph configuration. `TARGET_ATTENTION_BACKEND=triton` reproduces the
earlier target-backend setting in a fresh phase. Matched long-output probes
found FlashInfer faster for both variants, so `formal_flashinfer` is the final
matrix; the interrupted `formal` phase is retained as diagnostic evidence.

The LatentSpec BF16 token-cache path now fuses both stages' residual embedding
copies and cached QKV/RoPE into one Triton kernel. Set
`SGLANG_LATENTSPEC_FUSED_CACHED_INIT=0` to reproduce the previous three-kernel
path. The production kernel is checked against that path bit for bit, including
CUDA Graph replay after changing input ids and positions:

```bash
.venv/bin/python rebuttal/probe_init_fusion.py --production
.venv/bin/python -m unittest discover -s rebuttal -p test_benchmark_contract.py -v
```

Primary throughput is total successful completion tokens divided by the entire
measured request-group wall time, including prefill, network delivery and the
draining tail. The reference script's interval-union TPS is saved separately;
it can be higher because idle gaps are excluded. TTFT and TPOT are measured
from streaming text events; a speculative event can contain multiple tokens.
Fixed length uses `ignore_eos=true` without custom stops. Each request's actual
token count must equal the requested length, otherwise the case is invalid and
the matrix stops. Every cell starts with a cache flush. Server warmup uses 16
concurrent requests round-robin across datasets with 128 output tokens. Each
case also warms its dataset/batch shape with 128 tokens, then flushes the cache
again before timing. All warmup is excluded from measurements.

`stage_models.py` copies the S3-mounted checkpoints into
`/dev/shm/rebuttal-models` and checks SHA256 while copying and reading back.
After its manifest is complete, the launcher uses these identical local copies
to avoid repeatedly loading from object storage. Original paths and digests
are preserved in `output/model-cache-manifest.json`. Set `USE_MODEL_CACHE=0`
to load from the original paths instead. Model loading is outside timed cases.

`output/<phase>/<variant>/` contains metadata with dataset SHA256 hashes,
per-request timings, cell summaries, server logs, launch environment and a
framework patch. GPU driver/commit are recorded by the matrix runner.
Keep phase names unique when changing settings or framework code: resume only
applies to the exact same experiment configuration.

On the authenticated local host, run `python3 rebuttal/sync_feishu.py` to append
each result to the requested Feishu document and read it back for verification.
The synchronizer reconciles uncertain writes by case ID. It stores an ack for
each verified case in `sync_state.json`, retries transient failures, and keeps
all benchmark files independent of document availability.

After the formal run finishes, require the complete 150-cell paired comparison:

```bash
.venv/bin/python rebuttal/compare.py \
  --baseline rebuttal/output/formal_flashinfer/dspark \
  --candidate rebuttal/output/formal_flashinfer/dspark_latentspec \
  --output rebuttal/output/formal_flashinfer/comparison --require-complete
# Plot with a separate environment containing matplotlib, preserving serving dependencies.
.venv/bin/python -m venv rebuttal/.plot-venv
rebuttal/.plot-venv/bin/python -m pip install matplotlib
rebuttal/.plot-venv/bin/python rebuttal/plot_results.py \
  --comparison rebuttal/output/formal_flashinfer/comparison/comparison.json \
  --output rebuttal/output/formal_flashinfer/comparison
```

CSV, JSON and Feishu-ready XML tables are produced by `compare.py`. PNG/PDF/SVG
ratio plots explicitly state that each cell is one measurement without a
confidence interval. Tuning phases are separate from the formal 300-cell run.
