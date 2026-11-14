"""Isolated candidate: fuse the three cached LatentSpec initialization kernels.

This file does not alter serving defaults. Check bitwise parity and CUDA-graph
latency before considering integration into the framework.
"""
import json
import argparse
from pathlib import Path
import torch
import triton
import triton.language as tl
from sglang.kernels.ops.speculative.dspark.latent_init import (
    cached_qkv_rope, latent_proposal_init_residual,
)


@triton.jit
def _fused_init(ids, embedding, latent_cache, proposal_cache, latent_cos, proposal_cos,
                latent_pos, proposal_pos, latent_res, proposal_res, latent_qkv, proposal_qkv,
                EMBED_STRIDE: tl.constexpr, CACHE_STRIDE: tl.constexpr,
                D: tl.constexpr, Q: tl.constexpr, K: tl.constexpr,
                NQ: tl.constexpr, NK: tl.constexpr, HD: tl.constexpr,
                NL: tl.constexpr, NP: tl.constexpr, LATENT_ID: tl.constexpr,
                RES_CHUNK: tl.constexpr, RES_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    batch = row // (NL + NP)
    local = row % (NL + NP)
    latent = local < NL
    stage_row = tl.where(latent, local, local - NL)
    input_col = tl.where(latent, 0, stage_row)
    token = tl.load(ids + batch * NP + input_col)
    token = tl.where(latent & (stage_row > 0), LATENT_ID, token).to(tl.int64)
    out_row = batch * tl.where(latent, NL, NP) + stage_row
    cache = tl.where(latent, latent_cache, proposal_cache) + token * CACHE_STRIDE
    output = tl.where(latent, latent_qkv, proposal_qkv) + out_row * (Q + 2 * K)
    offsets = tl.arange(0, HD // 2)
    is_q = h < NQ
    head = tl.where(is_q, h, h - NQ)
    col = tl.where(is_q, head * HD, Q + head * HD)
    x1 = tl.load(cache + col + offsets).to(tl.float32)
    x2 = tl.load(cache + col + HD // 2 + offsets).to(tl.float32)
    position = tl.load(tl.where(latent, latent_pos, proposal_pos) + out_row).to(tl.int64)
    table = tl.where(latent, latent_cos, proposal_cos) + position * HD
    cos = tl.load(table + offsets).to(tl.float32)
    sin = tl.load(table + HD // 2 + offsets).to(tl.float32)
    tl.store(output + col + offsets, (x1 * cos - x2 * sin).to(tl.bfloat16))
    tl.store(output + col + HD // 2 + offsets, (x2 * cos + x1 * sin).to(tl.bfloat16))
    v_chunk: tl.constexpr = tl.cdiv(K, NQ)
    v_offset = h * v_chunk + offsets
    v_mask = is_q & (offsets < v_chunk) & (v_offset < K)
    v_col = Q + K + v_offset
    values = tl.load(cache + v_col, mask=v_mask)
    tl.store(output + v_col, values, mask=v_mask)
    e_offsets = h * RES_CHUNK + tl.arange(0, RES_BLOCK)
    e_mask = is_q & (tl.arange(0, RES_BLOCK) < RES_CHUNK) & (e_offsets < D)
    residual = tl.where(latent, latent_res, proposal_res) + out_row * D
    embedded = tl.load(embedding + token * EMBED_STRIDE + e_offsets, mask=e_mask)
    tl.store(residual + e_offsets, embedded, mask=e_mask)


def fused(input_ids, embedding, latent_cache, proposal_cache, cos, lp, pp,
          *, batch, nq, nk, hd, nl, np, latent_id):
    d = embedding.shape[1]
    width = (nq + 2 * nk) * hd
    lr = embedding.new_empty((batch * nl, d))
    pr = embedding.new_empty((batch * np, d))
    lq = embedding.new_empty((batch * nl, width))
    pq = embedding.new_empty((batch * np, width))
    chunk = triton.cdiv(d, nq)
    _fused_init[(batch * (nl + np), nq + nk)](
        input_ids, embedding, latent_cache, proposal_cache, cos, cos, lp, pp, lr, pr, lq, pq,
        EMBED_STRIDE=embedding.stride(0), CACHE_STRIDE=latent_cache.stride(0),
        D=d, Q=nq*hd, K=nk*hd, NQ=nq, NK=nk, HD=hd, NL=nl, NP=np, LATENT_ID=latent_id,
        RES_CHUNK=chunk, RES_BLOCK=triton.next_power_of_2(chunk), num_warps=2)
    return lr, pr, lq, pq


def main(production=False):
    torch.manual_seed(42)
    nq, nk, hd, d, nl, np, vocab = 32, 8, 128, 2560, 4, 7, 4096
    embedding = torch.randn((vocab, d), device="cuda", dtype=torch.bfloat16)
    lc = torch.randn((vocab, (nq+2*nk)*hd), device="cuda", dtype=torch.bfloat16)
    pc = torch.randn_like(lc)
    cos = torch.randn((16384, hd), device="cuda", dtype=torch.float32 if production else torch.bfloat16)
    proposal_cos = torch.randn_like(cos) if production else cos
    rows = []
    for batch in [1, 2, 4, 8, 16, 32]:
        ids = torch.randint(vocab, (batch*np,), device="cuda")
        lp = torch.randint(16384, (batch*nl,), device="cuda")
        pp = torch.randint(16384, (batch*np,), device="cuda")
        def ref():
            lr, pr = latent_proposal_init_residual(ids, embedding, batch_size=batch,
                num_latent_tokens=nl, num_proposal_tokens=np, latent_token_id=vocab-1)
            lq = cached_qkv_rope(ids, lc, cos, lp, num_q_heads=nq, num_k_heads=nk,
                head_dim=hd, tokens_per_batch=nl, inputs_per_batch=np, latent_token_id=vocab-1)
            pq = cached_qkv_rope(ids, pc, proposal_cos, pp, num_q_heads=nq, num_k_heads=nk,
                head_dim=hd, tokens_per_batch=np, inputs_per_batch=np)
            return lr, pr, lq, pq
        def candidate():
            if production:
                from sglang.kernels.ops.speculative.dspark.latent_cached_init import cached_latent_proposal_init
                return cached_latent_proposal_init(
                    ids, embedding, lc, pc, cos, proposal_cos, lp, pp, batch_size=batch,
                    num_q_heads=nq, num_k_heads=nk, head_dim=hd,
                    num_latent_tokens=nl, num_proposal_tokens=np, latent_token_id=vocab-1)
            return fused(ids, embedding, lc, pc, cos, lp, pp, batch=batch, nq=nq, nk=nk,
                         hd=hd, nl=nl, np=np, latent_id=vocab-1)
        for a, b in zip(ref(), candidate()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replay_output = candidate()
        torch.cuda.current_stream().wait_stream(stream)
        ids.random_(vocab)
        lp.random_(16384)
        pp.random_(16384)
        graph.replay()
        for a, b in zip(ref(), replay_output):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        ref_us = triton.testing.do_bench_cudagraph(ref, rep=500) * 1000
        fused_us = triton.testing.do_bench_cudagraph(candidate, rep=500) * 1000
        row = dict(batch=batch, reference_us=ref_us, candidate_us=fused_us,
                   speedup=ref_us/fused_us, bitwise_parity=True, replay_parity=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
    dest = Path(__file__).resolve().parent / ("output/init-fusion-production-probe.json" if production else "output/init-fusion-probe.json")
    dest.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", action="store_true")
    main(parser.parse_args().production)
