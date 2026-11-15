#!/usr/bin/env python3
"""Stage immutable S3-mounted checkpoints in local shared memory, verify SHA256."""
import hashlib
import json
from pathlib import Path
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

DEST = Path("/dev/shm/rebuttal-models")
OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    "target": Path("/mydata/Qwen/Qwen3-4B"),
    "dspark": Path("/mydata/models/qwen3-4b/dspark/step_2616"),
    "dspark_latentspec": Path("/mydata/models/qwen3-4b/dspark_latentspec/step_2616"),
}
DEST.mkdir(exist_ok=True)
needed = sum(f.stat().st_size for p in SOURCES.values() for f in p.iterdir() if f.is_file())
if shutil.disk_usage(DEST).free < needed + 1024**3:
    raise RuntimeError("Insufficient shared-memory space")
def copy_file(name, src):
    dest = DEST / name
    dest.mkdir(exist_ok=True)
    start = time.monotonic()
    out = dest / src.name
    tmp = out.with_suffix(out.suffix + ".tmp")
    digest = hashlib.sha256()
    print("COPY " + str(src), flush=True)
    with src.open("rb") as fi, tmp.open("wb") as fo:
        while block := fi.read(8 * 1024**2):
            fo.write(block)
            digest.update(block)
    if tmp.stat().st_size != src.stat().st_size:
        raise RuntimeError("Size mismatch: " + str(src))
    verified = hashlib.sha256()
    with tmp.open("rb") as fi:
        while block := fi.read(8 * 1024**2):
            verified.update(block)
    if verified.hexdigest() != digest.hexdigest():
        raise RuntimeError("SHA mismatch: " + str(src))
    tmp.replace(out)
    print(f"VERIFIED {out} {out.stat().st_size} bytes {time.monotonic()-start:.1f}s", flush=True)
    return str(out), {"source": str(src), "bytes": out.stat().st_size, "sha256": digest.hexdigest()}

manifest = {}
with ThreadPoolExecutor(max_workers=6) as pool:
    futures = [pool.submit(copy_file, name, src) for name, source in SOURCES.items()
               for src in sorted(source.iterdir()) if src.is_file()]
    for future in as_completed(futures):
        out, result = future.result()
        manifest[out] = result
(DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))
(OUTPUT / "model-cache-manifest.json").write_text(json.dumps(manifest, indent=2))
print("STAGING_COMPLETE", flush=True)
