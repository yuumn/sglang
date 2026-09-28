import torch
import triton
import triton.language as tl


@triton.jit
def _latent_init_rmsnorm_kernel(
    anchors_ptr,
    latent_token_ptr,
    weight_ptr,
    residual_ptr,
    normed_ptr,
    anchor_stride,
    D: tl.constexpr,
    NUM_LATENT: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    latent_col = row % NUM_LATENT
    batch = row // NUM_LATENT
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    source = tl.where(latent_col == 0, anchors_ptr + batch * anchor_stride, latent_token_ptr)
    values = tl.load(source + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / D + EPS)
    tl.store(residual_ptr + row * D + offsets, values, mask=mask)
    tl.store(normed_ptr + row * D + offsets, values * inv_rms * weight, mask=mask)


def latent_init_rmsnorm(
    anchors: torch.Tensor,
    latent_token: torch.Tensor,
    weight: torch.Tensor,
    *,
    num_latent_tokens: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``[anchor, latent, ...]`` and normalize it in one launch.

    The returned pair is ``(normalized, residual)``. Keeping the raw rows as
    the residual exactly matches the first pre-norm decoder layer while
    avoiding a separate concat/copy kernel before RMSNorm.
    """
    if anchors.ndim != 2 or latent_token.ndim != 1:
        raise ValueError(
            f"Expected anchors [B, D] and latent_token [D], got "
            f"{tuple(anchors.shape)} and {tuple(latent_token.shape)}."
        )
    batch_size, hidden_size = anchors.shape
    if latent_token.numel() != hidden_size or weight.numel() != hidden_size:
        raise ValueError("LatentSpec init tensors must use one common hidden size.")
    if anchors.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported LatentSpec activation dtype: {anchors.dtype}.")
    residual = torch.empty(
        (batch_size * num_latent_tokens, hidden_size),
        dtype=anchors.dtype,
        device=anchors.device,
    )
    normalized = torch.empty_like(residual)
    block_d = triton.next_power_of_2(hidden_size)
    num_warps = 8 if block_d >= 2048 else 4
    _latent_init_rmsnorm_kernel[(batch_size * num_latent_tokens,)](
        anchors,
        latent_token,
        weight,
        residual,
        normalized,
        anchors.stride(0),
        D=hidden_size,
        NUM_LATENT=num_latent_tokens,
        EPS=eps,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return normalized, residual


@triton.jit
def _latent_proposal_init_rmsnorm_kernel(
    input_ids_ptr,
    embedding_ptr,
    latent_weight_ptr,
    proposal_weight_ptr,
    latent_residual_ptr,
    latent_normed_ptr,
    proposal_residual_ptr,
    proposal_normed_ptr,
    embedding_stride,
    D: tl.constexpr,
    NUM_LATENT: tl.constexpr,
    NUM_PROPOSAL: tl.constexpr,
    LATENT_TOKEN_ID: tl.constexpr,
    LATENT_EPS: tl.constexpr,
    PROPOSAL_EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program = tl.program_id(0)
    rows_per_batch: tl.constexpr = NUM_LATENT + NUM_PROPOSAL
    batch = program // rows_per_batch
    local_row = program - batch * rows_per_batch
    is_latent = local_row < NUM_LATENT
    row = tl.where(is_latent, local_row, local_row - NUM_LATENT)

    # The first latent row is the proposal anchor. The remaining latent rows
    # use the trained latent token, while proposal rows use their actual ids
    # (anchor followed by masks in the DSpark serving path).
    input_col = tl.where(is_latent, 0, row)
    input_token = tl.load(input_ids_ptr + batch * NUM_PROPOSAL + input_col)
    source_token = tl.where(
        is_latent & (row > 0), LATENT_TOKEN_ID, input_token
    ).to(tl.int64)

    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    values = tl.load(
        embedding_ptr + source_token * embedding_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight_ptr = tl.where(
        is_latent, latent_weight_ptr, proposal_weight_ptr
    )
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    eps = tl.where(is_latent, LATENT_EPS, PROPOSAL_EPS)
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / D + eps)

    residual_ptr = tl.where(
        is_latent,
        latent_residual_ptr + (batch * NUM_LATENT + row) * D,
        proposal_residual_ptr + (batch * NUM_PROPOSAL + row) * D,
    )
    normed_ptr = tl.where(
        is_latent,
        latent_normed_ptr + (batch * NUM_LATENT + row) * D,
        proposal_normed_ptr + (batch * NUM_PROPOSAL + row) * D,
    )
    tl.store(residual_ptr + offsets, values, mask=mask)
    tl.store(normed_ptr + offsets, values * inv_rms * weight, mask=mask)


def latent_proposal_init_rmsnorm(
    input_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    latent_norm_weight: torch.Tensor,
    proposal_norm_weight: torch.Tensor,
    *,
    batch_size: int,
    num_latent_tokens: int,
    num_proposal_tokens: int,
    latent_token_id: int,
    latent_eps: float,
    proposal_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed and pre-normalize both trained LatentSpec stages in one launch."""
    hidden_size = embedding_weight.shape[1]
    if input_ids.numel() != batch_size * num_proposal_tokens:
        raise ValueError(
            f"Expected {batch_size * num_proposal_tokens} proposal ids, "
            f"got {input_ids.numel()}."
        )
    if embedding_weight.ndim != 2 or embedding_weight.stride(1) != 1:
        raise ValueError("LatentSpec fused embedding requires row-major weights.")
    if embedding_weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"Unsupported LatentSpec embedding dtype: {embedding_weight.dtype}."
        )
    shape = (batch_size, hidden_size)
    latent_residual = torch.empty(
        (shape[0] * num_latent_tokens, shape[1]),
        dtype=embedding_weight.dtype,
        device=embedding_weight.device,
    )
    latent_normed = torch.empty_like(latent_residual)
    proposal_residual = torch.empty(
        (shape[0] * num_proposal_tokens, shape[1]),
        dtype=embedding_weight.dtype,
        device=embedding_weight.device,
    )
    proposal_normed = torch.empty_like(proposal_residual)
    block_d = triton.next_power_of_2(hidden_size)
    num_warps = 8 if block_d >= 2048 else 4
    grid = (batch_size * (num_latent_tokens + num_proposal_tokens),)
    _latent_proposal_init_rmsnorm_kernel[grid](
        input_ids,
        embedding_weight,
        latent_norm_weight,
        proposal_norm_weight,
        latent_residual,
        latent_normed,
        proposal_residual,
        proposal_normed,
        embedding_weight.stride(0),
        D=hidden_size,
        NUM_LATENT=num_latent_tokens,
        NUM_PROPOSAL=num_proposal_tokens,
        LATENT_TOKEN_ID=latent_token_id,
        LATENT_EPS=latent_eps,
        PROPOSAL_EPS=proposal_eps,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return latent_normed, latent_residual, proposal_normed, proposal_residual


@triton.jit
def _cached_qkv_rope_kernel(
    input_ids_ptr,
    cache_ptr,
    cos_sin_ptr,
    positions_ptr,
    output_ptr,
    cache_stride,
    output_stride,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOKENS_PER_BATCH: tl.constexpr,
    INPUTS_PER_BATCH: tl.constexpr,
    LATENT_TOKEN_ID: tl.constexpr,
    LATENT_LAYOUT: tl.constexpr,
):
    token_row = tl.program_id(0)
    head_program = tl.program_id(1)
    if LATENT_LAYOUT:
        batch = token_row // TOKENS_PER_BATCH
        local_row = token_row - batch * TOKENS_PER_BATCH
        token_id = tl.where(
            local_row == 0,
            tl.load(input_ids_ptr + batch * INPUTS_PER_BATCH),
            LATENT_TOKEN_ID,
        ).to(tl.int64)
    else:
        token_id = tl.load(input_ids_ptr + token_row).to(tl.int64)

    source = cache_ptr + token_id * cache_stride
    output = output_ptr + token_row * output_stride
    half: tl.constexpr = HEAD_DIM // 2
    offsets = tl.arange(0, half)
    is_q = head_program < NUM_Q_HEADS
    head = tl.where(is_q, head_program, head_program - NUM_Q_HEADS)
    col0 = tl.where(is_q, head * HEAD_DIM, Q_SIZE + head * HEAD_DIM)
    x1 = tl.load(source + col0 + offsets).to(tl.float32)
    x2 = tl.load(source + col0 + half + offsets).to(tl.float32)
    position = tl.load(positions_ptr + token_row).to(tl.int64)
    cos = tl.load(cos_sin_ptr + position * HEAD_DIM + offsets).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + position * HEAD_DIM + half + offsets
    ).to(tl.float32)
    tl.store(
        output + col0 + offsets,
        (x1 * cos - x2 * sin).to(tl.bfloat16),
    )
    tl.store(
        output + col0 + half + offsets,
        (x2 * cos + x1 * sin).to(tl.bfloat16),
    )

    # Q has more heads than V has 64-element chunks for Qwen3.  Let each
    # Q-head CTA copy one disjoint V slice while it is already resident,
    # avoiding a separate launch or extra V-only CTAs.
    v_per_q_head: tl.constexpr = tl.cdiv(KV_SIZE, NUM_Q_HEADS)
    v_offsets = head_program * v_per_q_head + offsets
    v_mask = is_q & (offsets < v_per_q_head) & (v_offsets < KV_SIZE)
    v_col = Q_SIZE + KV_SIZE + v_offsets
    values = tl.load(source + v_col, mask=v_mask)
    tl.store(output + v_col, values, mask=v_mask)


@triton.jit
def _cached_v_gather_kernel(
    input_ids_ptr,
    cache_ptr,
    output_ptr,
    cache_stride,
    output_stride,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    TOKENS_PER_BATCH: tl.constexpr,
    INPUTS_PER_BATCH: tl.constexpr,
    LATENT_TOKEN_ID: tl.constexpr,
    LATENT_LAYOUT: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    token_row = tl.program_id(0)
    if LATENT_LAYOUT:
        batch = token_row // TOKENS_PER_BATCH
        local_row = token_row - batch * TOKENS_PER_BATCH
        token_id = tl.where(
            local_row == 0,
            tl.load(input_ids_ptr + batch * INPUTS_PER_BATCH),
            LATENT_TOKEN_ID,
        ).to(tl.int64)
    else:
        token_id = tl.load(input_ids_ptr + token_row).to(tl.int64)
    offsets = tl.arange(0, BLOCK_V)
    mask = offsets < KV_SIZE
    v_col = Q_SIZE + KV_SIZE + offsets
    source = cache_ptr + token_id * cache_stride
    output = output_ptr + token_row * output_stride
    values = tl.load(source + v_col, mask=mask)
    tl.store(output + v_col, values, mask=mask)


def cached_qkv_rope(
    input_ids: torch.Tensor,
    normalized_qkv_cache: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
    tokens_per_batch: int,
    inputs_per_batch: int,
    latent_token_id: int = -1,
) -> torch.Tensor:
    """Gather cached first-layer Q/K/V and apply position-dependent RoPE.

    Q and K in ``normalized_qkv_cache`` already include the trained per-head
    RMSNorm.  The only request-dependent work is selecting the token row and
    applying RoPE, which this single kernel performs without a small GEMM.
    ``latent_token_id >= 0`` selects the ``[anchor, latent, ...]`` layout;
    otherwise input ids are consumed row-for-row for the proposal layout.
    """
    q_size = num_q_heads * head_dim
    kv_size = num_k_heads * head_dim
    expected_width = q_size + 2 * kv_size
    if normalized_qkv_cache.ndim != 2 or normalized_qkv_cache.shape[1] != expected_width:
        raise ValueError(
            "Unexpected cached QKV shape: "
            f"{tuple(normalized_qkv_cache.shape)}, expected width {expected_width}."
        )
    num_tokens = positions.numel()
    if num_tokens == 0:
        return normalized_qkv_cache.new_empty((0, expected_width))
    output = normalized_qkv_cache.new_empty((num_tokens, expected_width))
    _cached_qkv_rope_kernel[
        (num_tokens, num_q_heads + num_k_heads)
    ](
        input_ids,
        normalized_qkv_cache,
        cos_sin_cache,
        positions,
        output,
        normalized_qkv_cache.stride(0),
        output.stride(0),
        Q_SIZE=q_size,
        KV_SIZE=kv_size,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        HEAD_DIM=head_dim,
        TOKENS_PER_BATCH=tokens_per_batch,
        INPUTS_PER_BATCH=inputs_per_batch,
        LATENT_TOKEN_ID=latent_token_id,
        LATENT_LAYOUT=latent_token_id >= 0,
        num_warps=2,
    )
    return output


@triton.jit
def _latent_proposal_init_residual_kernel(
    input_ids_ptr,
    embedding_ptr,
    latent_residual_ptr,
    proposal_residual_ptr,
    embedding_stride,
    D: tl.constexpr,
    NUM_LATENT: tl.constexpr,
    NUM_PROPOSAL: tl.constexpr,
    LATENT_TOKEN_ID: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program = tl.program_id(0)
    rows_per_batch: tl.constexpr = NUM_LATENT + NUM_PROPOSAL
    batch = program // rows_per_batch
    local_row = program - batch * rows_per_batch
    is_latent = local_row < NUM_LATENT
    row = tl.where(is_latent, local_row, local_row - NUM_LATENT)
    input_col = tl.where(is_latent, 0, row)
    input_token = tl.load(input_ids_ptr + batch * NUM_PROPOSAL + input_col)
    source_token = tl.where(
        is_latent & (row > 0), LATENT_TOKEN_ID, input_token
    ).to(tl.int64)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    values = tl.load(
        embedding_ptr + source_token * embedding_stride + offsets,
        mask=mask,
    )
    destination = tl.where(
        is_latent,
        latent_residual_ptr + (batch * NUM_LATENT + row) * D,
        proposal_residual_ptr + (batch * NUM_PROPOSAL + row) * D,
    )
    tl.store(destination + offsets, values, mask=mask)


def latent_proposal_init_residual(
    input_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    *,
    batch_size: int,
    num_latent_tokens: int,
    num_proposal_tokens: int,
    latent_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the two first-layer residual rails when QKV is token-cached."""
    hidden_size = embedding_weight.shape[1]
    latent = embedding_weight.new_empty(
        (batch_size * num_latent_tokens, hidden_size)
    )
    proposal = embedding_weight.new_empty(
        (batch_size * num_proposal_tokens, hidden_size)
    )
    block_d = triton.next_power_of_2(hidden_size)
    _latent_proposal_init_residual_kernel[
        (batch_size * (num_latent_tokens + num_proposal_tokens),)
    ](
        input_ids,
        embedding_weight,
        latent,
        proposal,
        embedding_weight.stride(0),
        D=hidden_size,
        NUM_LATENT=num_latent_tokens,
        NUM_PROPOSAL=num_proposal_tokens,
        LATENT_TOKEN_ID=latent_token_id,
        BLOCK_D=block_d,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return latent, proposal
