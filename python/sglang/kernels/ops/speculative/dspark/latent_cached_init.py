# SPDX-License-Identifier: Apache-2.0
"""Fuse cached LatentSpec residual initialization and first-layer QKV/RoPE.

The cache already contains the trained Q/K normalization. Both stages select
and rotate the same rows as cached_qkv_rope, while Q-head blocks also copy
one disjoint embedding slice into each stage's residual rail.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _cached_latent_proposal_init_kernel(ids, embedding, latent_cache, proposal_cache, latent_cos, proposal_cos,
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


def cached_latent_proposal_init(
    input_ids: torch.Tensor,
    embedding: torch.Tensor,
    latent_cache: torch.Tensor,
    proposal_cache: torch.Tensor,
    latent_cos: torch.Tensor,
    proposal_cos: torch.Tensor,
    latent_positions: torch.Tensor,
    proposal_positions: torch.Tensor,
    *,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
    num_latent_tokens: int,
    num_proposal_tokens: int,
    latent_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return latent/proposal residuals and position-dependent packed QKV."""
    if (
        embedding.dtype != torch.bfloat16
        or latent_cache.dtype != embedding.dtype
        or proposal_cache.dtype != embedding.dtype
    ):
        raise ValueError("Cached LatentSpec initialization requires BF16 tensors")
    if (
        latent_cache.shape != proposal_cache.shape
        or latent_cache.stride() != proposal_cache.stride()
    ):
        raise ValueError("The two first-layer caches must share shape and strides")
    if latent_cos.dtype != proposal_cos.dtype:
        raise ValueError("The two RoPE tables must have the same dtype")
    if (
        input_ids.numel() != batch_size * num_proposal_tokens
        or latent_positions.numel() != batch_size * num_latent_tokens
        or proposal_positions.numel() != batch_size * num_proposal_tokens
    ):
        raise ValueError("Input ids and stage positions do not match the batch layout")
    if triton.cdiv(num_k_heads * head_dim, num_q_heads) > head_dim // 2:
        raise ValueError("Unsupported Q/K head ratio for fused cached initialization")
    hidden_size = embedding.shape[1]
    width = (num_q_heads + 2 * num_k_heads) * head_dim
    if latent_cache.shape[1] != width:
        raise ValueError("Cached QKV width does not match the attention heads")
    latent_residual = embedding.new_empty((batch_size * num_latent_tokens, hidden_size))
    proposal_residual = embedding.new_empty((batch_size * num_proposal_tokens, hidden_size))
    latent_qkv = embedding.new_empty((batch_size * num_latent_tokens, width))
    proposal_qkv = embedding.new_empty((batch_size * num_proposal_tokens, width))
    chunk = triton.cdiv(hidden_size, num_q_heads)
    grid = (batch_size * (num_latent_tokens + num_proposal_tokens), num_q_heads + num_k_heads)
    _cached_latent_proposal_init_kernel[grid](
        input_ids, embedding, latent_cache, proposal_cache, latent_cos, proposal_cos,
        latent_positions, proposal_positions, latent_residual, proposal_residual,
        latent_qkv, proposal_qkv,
        EMBED_STRIDE=embedding.stride(0), CACHE_STRIDE=latent_cache.stride(0),
        D=hidden_size, Q=num_q_heads * head_dim, K=num_k_heads * head_dim,
        NQ=num_q_heads, NK=num_k_heads, HD=head_dim, NL=num_latent_tokens,
        NP=num_proposal_tokens, LATENT_ID=latent_token_id,
        RES_CHUNK=chunk, RES_BLOCK=triton.next_power_of_2(chunk), num_warps=2,
    )
    return latent_residual, proposal_residual, latent_qkv, proposal_qkv

