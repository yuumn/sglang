#!/usr/bin/env python3
"""Check actual chat-template token counts for every full-dataset input on CPU."""
import json
from pathlib import Path

from full_common import ROOT, DATASETS, COUNTS
from full_suite import TARGETS
import sys
sys.path.insert(0, str(ROOT / "latentspec_scripts"))
from test_tps import resolve_datasets
from transformers import AutoTokenizer


prompts, _ = resolve_datasets(ROOT / "datasets", DATASETS, 0)
result = {}
for family, source in TARGETS.items():
    if family == "qwen3-4b" and Path("/dev/shm/rebuttal-models/target").exists():
        source = "/dev/shm/rebuttal-models/target"
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=True)
    result[family] = {}
    for name, selected in prompts.items():
        encoded = tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}] for prompt in selected],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True)
        lengths = [len(row) for row in encoded["input_ids"]]
        assert len(lengths) == len(selected) == COUNTS[name]
        result[family][name] = {
            "count": len(lengths), "max_input_tokens": max(lengths),
            "max_with_fixed4096": max(lengths) + 4096,
            "fits_context16384": max(lengths) + 4096 <= 16384}
    print(json.dumps({family: result[family]}), flush=True)
dest = ROOT / "rebuttal/output/full-context-preflight.json"
dest.write_text(json.dumps(result, indent=2))
assert all(row["fits_context16384"] for family in result.values() for row in family.values())
