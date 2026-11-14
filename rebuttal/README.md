# H20 Qwen3-4B throughput comparison

Run on `h20` from `/root/workspace/sglang`, using `.venv`.

```bash
rebuttal/launch_server.sh dspark
rebuttal/launch_server.sh dspark_latentspec

# Owns and stops each server process group; do not run another GPU workload.
.venv/bin/python -u rebuttal/run_matrix.py --phase baseline --requests 32

# Resume: repeat the same command. Valid completed cells are skipped.
# --requests 0 uses all dataset prompts; --lengths 0 means natural EOS, cap 512.
.venv/bin/python rebuttal/status.py --phase formal
.venv/bin/python rebuttal/compare.py \
  --baseline rebuttal/output/baseline/dspark \
  --candidate rebuttal/output/baseline/dspark_latentspec \
  --output rebuttal/output/baseline/comparison
```

The default matrix is six datasets × five concurrencies × five output policies
(natural EOS with cap 512, fixed 512/1024/2048/4096) × two variants = 300 cells.
Prompts are the same first 32 entries (AIME25 has 30). Sampling follows the
reference script: temperature 1, top_p 1, seed 980406, thinking disabled.
Both models use gamma 7 and static full verification. Target and draft attention
use Triton, BF16 weights/KV, TP=1 and the same CUDA graph configuration.

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
  --baseline rebuttal/output/formal/dspark \
  --candidate rebuttal/output/formal/dspark_latentspec \
  --output rebuttal/output/formal/comparison --require-complete
# Plot with a separate environment containing matplotlib, preserving serving dependencies.
.venv/bin/python -m venv rebuttal/.plot-venv
rebuttal/.plot-venv/bin/python -m pip install matplotlib
rebuttal/.plot-venv/bin/python rebuttal/plot_results.py \
  --comparison rebuttal/output/formal/comparison/comparison.json \
  --output rebuttal/output/formal/comparison
```

CSV, JSON and Feishu-ready XML tables are produced by `compare.py`. PNG/PDF/SVG
ratio plots explicitly state that each cell is one measurement without a
confidence interval. Tuning phases are separate from the formal 300-cell run.
