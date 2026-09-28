"""SGLang inference model for the Qwen3 MySpec/LatentSpec checkpoint.

The model keeps the two trained static stages distinct: the latent layers run
four rows per request and the proposal layers run seven.  Both stages use
batched RadixAttention and dedicated capture-stable metadata, so preserving
the training layout does not require giving up CUDA graphs.
"""

from __future__ import annotations

import logging
from copy import copy
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashAttention, DFlashDecoderLayer
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

logger = logging.getLogger(__name__)


def _linear(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, module.weight, module.bias)


class _LatentSpecAttention(DFlashAttention):
    """DFlash cache plumbing plus the separate MySpec context projections."""

    def __init__(
        self, config, layer_id: int, *, with_latent_kv: bool, quant_config=None
    ):
        super().__init__(
            config=config, layer_id=layer_id, quant_config=quant_config
        )
        hidden = int(config.hidden_size)
        kv_out = self.total_num_kv_heads * self.head_dim
        self.kv_out = kv_out
        bias = bool(getattr(config, "attention_bias", False))
        # K and V are independent in the checkpoint, but evaluating them as one
        # packed projection removes two small-GEMM launches from every layer.
        # The loader below copies the original tensors into the matching halves.
        self.kv_ctx = nn.Linear(hidden, 2 * kv_out, bias=bias)
        self.kv_latent = (
            nn.Linear(hidden, 2 * kv_out, bias=bias) if with_latent_kv else None
        )

    def kv_proj_only(self, hidden_states: torch.Tensor):
        return _linear(self.kv_ctx, hidden_states).split(self.kv_out, dim=-1)

    @property
    def context_kv_weight(self) -> torch.Tensor:
        return self.kv_ctx.weight

    def project_current(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        if self.use_table_qk_norm_rope and qkv.dtype == torch.bfloat16:
            from sglang.srt.speculative.dflash_utils import table_qk_norm_rope_

            table_qk_norm_rope_(
                qkv,
                positions,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.q_norm.variance_epsilon,
            )
            return qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )

        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q = self.q_norm(q.reshape(-1, self.head_dim)).view_as(q)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view_as(k)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def prepare_latent_kv(
        self, packed_kv: torch.Tensor, positions: torch.Tensor
    ):
        if self.use_table_qk_norm_rope and packed_kv.dtype == torch.bfloat16:
            from sglang.srt.speculative.dflash_utils import table_k_norm_rope_

            table_k_norm_rope_(
                packed_kv,
                positions,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                self.num_kv_heads,
                self.head_dim,
                self.k_norm.variance_epsilon,
            )
            return packed_kv.split(self.kv_out, dim=-1)

        k, v = packed_kv.split(self.kv_out, dim=-1)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view_as(k)
        k = self.apply_k_rope(positions, k)
        return k, v

    def project_latent_kv(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ):
        assert self.kv_latent is not None
        return self.prepare_latent_kv(
            _linear(self.kv_latent, hidden_states), positions
        )


class _LatentSpecLayer(DFlashDecoderLayer):
    def __init__(
        self, config, layer_id: int, *, with_latent_kv: bool, quant_config=None
    ):
        nn.Module.__init__(self)
        hidden = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.self_attn = _LatentSpecAttention(
            config,
            layer_id=layer_id,
            with_latent_kv=with_latent_kv,
            quant_config=quant_config,
        )
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)
        # Reuse the SGLang Qwen3-compatible gated MLP and its weight loader.
        from sglang.srt.models.dflash import DFlashMLP

        self.mlp = DFlashMLP(config, quant_config=quant_config)
        self.attention_conv = None
        self.mlp_conv = None


class _LatentSpecBackbone(nn.Module):
    """Native loader/executor for ``Qwen3MySpecModel`` checkpoints.

    TP=1 is intentional for the initial native implementation.  Target
    inference may still use arbitrary supported parallelism.
    """

    supports_fused_context_kv = True
    candidate_selector = None
    is_nemotron_35_draft = False

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        del prefix
        self.config = config
        self.draft_config = parse_dflash_draft_config(draft_hf_config=config)
        self.proposal_size = int(config.block_size)
        self.num_latent_tokens = int(config.num_latent_tokens)
        # The outer DFlash protocol only carries the seven proposal rows. The
        # first four cache locations double as the transient latent-K/V prefix;
        # target verification overwrites them after the draft forward.
        self.block_size = self.proposal_size
        self.num_latent_layers = int(config.num_latent_layers)
        self.num_proposal_layers = int(config.num_hidden_layers)
        self.latent_token_id = int(config.latent_token_id)
        self.mask_token_id = int(config.mask_token_id)
        hidden = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        # The checkpoint records layer_types for the proposal stack only;
        # SGLang allocates independent attention layer IDs for both stacks.
        config.layer_types = ["full_attention"] * (
            self.num_latent_layers + int(config.num_hidden_layers)
        )

        self.embed_tokens = nn.Embedding(int(config.vocab_size), hidden)
        self.fc = nn.Linear(len(config.target_layer_ids) * hidden, hidden, bias=False)
        self.hidden_norm = RMSNorm(hidden, eps=eps)
        self.latent_hidden_norm = RMSNorm(hidden, eps=eps)
        self.norm = RMSNorm(hidden, eps=eps)

        latent = [
            _LatentSpecLayer(
                config, i, with_latent_kv=False, quant_config=quant_config
            )
            for i in range(self.num_latent_layers)
        ]
        mask = [
            _LatentSpecLayer(
                config,
                self.num_latent_layers + i,
                with_latent_kv=True,
                quant_config=quant_config,
            )
            for i in range(int(config.num_hidden_layers))
        ]
        # DFlashWorker iterates `layers` to materialize context KV for every
        # attention layer. Keep one flat list and remember the stage boundary.
        self.layers = nn.ModuleList(latent + mask)
        # Populated once loading has filled the per-layer checkpoint weights.
        # One wider GEMM is substantially more efficient than three tiny
        # latent K/V projections in every draft replay.
        self.register_buffer(
            "_stacked_latent_kv_weight",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_stacked_latent_kv_bias", None, persistent=False
        )
        self.register_buffer(
            "_first_latent_token_qkv_cache", None, persistent=False
        )
        self.register_buffer(
            "_first_proposal_token_qkv_cache", None, persistent=False
        )
        self._fused_latent_kv_write_bundle = None
        # The first proposal QKV projection depends only on the fixed
        # anchor/mask input, not on the latent stack.  Run it on a side stream
        # while the two latent layers execute, then join immediately before
        # proposal attention consumes it.  The stream is created lazily during
        # graph warmup so CUDA graph capture records the same dependency.
        self._proposal_projection_stream = None

    def set_block_size(self, block_size: int) -> None:
        if int(block_size) != self.block_size:
            raise ValueError(
                "LATENTSPEC draft width is fixed by the trained checkpoint: "
                f"block_size={self.block_size}, "
                f"requested={block_size}."
            )

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_attention_sliding_window_size(self):
        return None

    def set_runtime_pools(self, *, token_to_kv_pool, req_to_token_pool) -> None:
        del req_to_token_pool
        if (
            envs.SGLANG_LATENTSPEC_TOKEN_QKV_CACHE.get()
            and self._first_latent_token_qkv_cache is None
        ):
            self._first_latent_token_qkv_cache = self._build_token_qkv_cache(
                self.layers[0]
            )
            self._first_proposal_token_qkv_cache = self._build_token_qkv_cache(
                self.layers[self.num_latent_layers]
            )
            torch.cuda.empty_cache()
            cache_gib = (
                self._first_latent_token_qkv_cache.numel()
                + self._first_proposal_token_qkv_cache.numel()
            ) * self._first_latent_token_qkv_cache.element_size() / (1024**3)
            logger.info(
                "LatentSpec BF16 first-layer token QKV caches enabled (%.2f GiB).",
                cache_gib,
            )
        self._fused_latent_kv_write_bundle = (
            self._build_fused_latent_kv_write_bundle(token_to_kv_pool)
            if envs.SGLANG_LATENTSPEC_FUSED_LATENT_KV.get()
            else None
        )
        if self._fused_latent_kv_write_bundle is not None:
            logger.info(
                "LatentSpec fused latent-KV normalization/RoPE/cache write enabled."
            )
        elif envs.SGLANG_LATENTSPEC_FUSED_LATENT_KV.get():
            logger.warning(
                "LatentSpec fused latent-KV write is incompatible with this KV "
                "pool; using the per-layer fallback."
            )

    @torch.inference_mode()
    def _build_token_qkv_cache(self, layer) -> torch.Tensor:
        """Precompute position-independent first-layer Q/K/V for every token."""
        embedding = self.embed_tokens.weight
        attn = layer.self_attn
        width = attn.q_size + 2 * attn.kv_size
        cache = torch.empty(
            (embedding.shape[0], width),
            dtype=embedding.dtype,
            device=embedding.device,
        )
        # Bound temporary memory while retaining a GEMM shape large enough for
        # efficient one-time cache construction.
        chunk_rows = 4096
        for start in range(0, embedding.shape[0], chunk_rows):
            end = min(start + chunk_rows, embedding.shape[0])
            normalized = layer.input_layernorm(embedding[start:end])
            qkv, _ = attn.qkv_proj(normalized)
            q, k, v = qkv.split(
                [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
            )
            q = attn.q_norm(q.reshape(-1, attn.head_dim)).view_as(q)
            k = attn.k_norm(k.reshape(-1, attn.head_dim)).view_as(k)
            cache[start:end, : attn.q_size].copy_(q)
            cache[
                start:end, attn.q_size : attn.q_size + attn.kv_size
            ].copy_(k)
            cache[start:end, attn.q_size + attn.kv_size :].copy_(v)
        return cache

    def _build_fused_latent_kv_write_bundle(self, pool):
        layers = list(self.layers[self.num_latent_layers :])
        if not layers or not (
            hasattr(pool, "get_key_buffer") and hasattr(pool, "get_value_buffer")
        ):
            return None
        attn0 = layers[0].self_attn
        head_dim = attn0.head_dim
        kv_size = attn0.kv_size
        rotary = attn0.rotary_emb
        if type(rotary).__name__ != "RotaryEmbedding":
            return None
        if getattr(rotary, "rotary_dim", None) != head_dim:
            return None
        eps = attn0.k_norm.variance_epsilon
        k_norm_weights, meta_rows = [], []
        for layer in layers:
            attn = layer.self_attn
            if attn.attn.k_scale is not None or attn.attn.v_scale is not None:
                return None
            if attn.head_dim != head_dim or attn.kv_size != kv_size:
                return None
            if attn.rotary_emb is not rotary and not torch.equal(
                attn.rotary_emb.cos_sin_cache, rotary.cos_sin_cache
            ):
                return None
            if attn.rotary_emb.is_neox_style != rotary.is_neox_style:
                return None
            if attn.k_norm.variance_epsilon != eps:
                return None
            k_buf = pool.get_key_buffer(attn.attn.layer_id)
            v_buf = pool.get_value_buffer(attn.attn.layer_id)
            num_kv_heads = kv_size // head_dim
            for buf in (k_buf, v_buf):
                if buf.dtype != torch.bfloat16:
                    return None
                if buf.shape[1:] != (num_kv_heads, head_dim):
                    return None
                if buf.stride(1) != head_dim or buf.stride(2) != 1:
                    return None
            k_norm_weights.append(attn.k_norm.weight.data)
            meta_rows.append(
                [k_buf.data_ptr(), v_buf.data_ptr(), k_buf.stride(0), v_buf.stride(0)]
            )
        device = k_norm_weights[0].device
        return (
            torch.tensor(meta_rows, dtype=torch.int64, device=device),
            torch.stack(k_norm_weights).to(device),
            rotary.cos_sin_cache.to(device),
            eps,
            len(layers),
            kv_size,
            head_dim,
            rotary.is_neox_style,
        )

    def _write_fused_latent_kv(
        self,
        packed_kv: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
    ) -> bool:
        bundle = self._fused_latent_kv_write_bundle
        if bundle is None:
            return False
        from sglang.kernels.ops.speculative.dspark.fused_kv_write import (
            fused_kv_norm_rope_write,
        )

        meta, k_norm, cos_sin, eps, num_layers, kv_size, head_dim, is_neox = bundle
        fused_kv_norm_rope_write(
            packed_kv,
            meta,
            k_norm,
            cos_sin,
            positions,
            cache_locs,
            num_layers,
            kv_size,
            head_dim,
            eps,
            is_neox_style=is_neox,
        )
        return True

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(target_hidden))

    def prepare_context_hidden_for_kv(self, layer, ctx_hidden):
        return ctx_hidden

    def _run_layer(
        self,
        layer,
        hidden,
        residual,
        positions,
        forward_batch: ForwardBatch,
        *,
        input_is_normalized: bool = False,
        projected_current=None,
    ):
        # Keep the same two-rail representation as the native DFlash/Qwen
        # layer: ``hidden`` is the latest branch output and ``residual`` is the
        # accumulated skip path.  Fusing their addition into the following
        # RMSNorm removes two full hidden-state reads/writes per layer.
        if input_is_normalized:
            if residual is None:
                raise ValueError("A pre-normalized LatentSpec layer needs its residual.")
        elif residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)
        if projected_current is None:
            q, k_cur, v_cur = layer.self_attn.project_current(hidden, positions)
        else:
            q, k_cur, v_cur = projected_current
        # Draft-local K/V is consumed directly by this call. Committed context
        # K/V is rematerialized from target hidden states after verification,
        # so writing these transient rows would only add cache traffic.
        attn_out = layer.self_attn.attn(
            q, k_cur, v_cur, forward_batch, save_kv_cache=False
        )
        attn_out, _ = layer.self_attn.o_proj(attn_out)
        hidden, residual = layer.post_attention_layernorm(attn_out, residual)
        return layer.mlp(hidden), residual

    @staticmethod
    def _set_stage(backend, stage: str) -> None:
        setter = getattr(backend, "set_latentspec_stage", None)
        if setter is None:
            raise RuntimeError(
                "LATENTSPEC two-stage execution requires a draft attention "
                "backend with set_latentspec_stage(); use "
                "--speculative-draft-attention-backend triton."
            )
        setter(stage)

    @staticmethod
    def _write_latent_kv(
        layer,
        latent,
        latent_positions,
        cache_locs,
        backend,
        projected_kv=None,
    ):
        attn = layer.self_attn
        if projected_kv is None:
            k_lat, v_lat = attn.project_latent_kv(latent, latent_positions)
        else:
            k_lat, v_lat = attn.prepare_latent_kv(
                projected_kv, latent_positions
            )
        k_lat = k_lat.view(-1, attn.num_kv_heads, attn.head_dim)
        v_lat = v_lat.view(-1, attn.num_kv_heads, attn.head_dim)
        backend.token_to_kv_pool.set_kv_buffer(
            attn.attn,
            cache_locs,
            k_lat,
            v_lat,
            attn.attn.k_scale,
            attn.attn.v_scale,
        )

    def _make_stage_batches(
        self,
        forward_batch: ForwardBatch,
        latent_positions: torch.Tensor,
        proposal_positions: torch.Tensor,
    ):
        bs = int(forward_batch.batch_size)
        cache_locs = forward_batch.out_cache_loc.view(bs, self.block_size)

        # contiguous() is intentional. During CUDA graph replay this tiny
        # gather copy is replayed from the freshly staged row-major outer
        # buffer, while latent K/V materialization receives dense cache locs.
        latent_cache_locs = (
            cache_locs[:, : self.num_latent_tokens].contiguous().view(-1)
        )
        latent_batch = copy(forward_batch)
        latent_batch.out_cache_loc = latent_cache_locs
        latent_batch.positions = latent_positions

        proposal_batch = copy(forward_batch)
        proposal_batch.positions = proposal_positions
        # The stage-specific Triton CSR metadata already adds the four latent
        # slots to the proposal prefix.  Keeping the shared sequence lengths
        # here avoids a redundant per-replay vector-add kernel; this batch is
        # only consumed by that LatentSpec-aware backend.
        return latent_batch, proposal_batch, latent_cache_locs

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        del kwargs
        bs = int(forward_batch.batch_size)
        raw_pos = positions.view(bs, self.block_size)
        latent_positions = raw_pos[:, : self.num_latent_tokens]
        latent_pos_flat = (
            latent_positions.reshape(-1)
            if latent_positions.is_contiguous()
            else latent_positions.contiguous().view(-1)
        )
        proposal_pos_flat = positions
        first_latent_layer = self.layers[0]
        first_proposal_layer = self.layers[self.num_latent_layers]
        proposal_is_normalized = input_embeds is None
        token_qkv_cached = (
            proposal_is_normalized
            and self._first_latent_token_qkv_cache is not None
            and self._first_proposal_token_qkv_cache is not None
        )
        if token_qkv_cached:
            from sglang.kernels.ops.speculative.dspark.latent_init import (
                latent_proposal_init_residual,
            )

            latent_residual, proposal_residual = latent_proposal_init_residual(
                input_ids,
                self.embed_tokens.weight,
                batch_size=bs,
                num_latent_tokens=self.num_latent_tokens,
                num_proposal_tokens=self.proposal_size,
                latent_token_id=self.latent_token_id,
            )
            # The cached QKV tensors already contain the first input RMSNorm;
            # only the raw embedding residual rails are needed here.
            latent = latent_residual
            proposal = proposal_residual
        elif proposal_is_normalized:
            from sglang.kernels.ops.speculative.dspark.latent_init import (
                latent_proposal_init_rmsnorm,
            )

            latent, latent_residual, proposal, proposal_residual = (
                latent_proposal_init_rmsnorm(
                    input_ids,
                    self.embed_tokens.weight,
                    first_latent_layer.input_layernorm.weight,
                    first_proposal_layer.input_layernorm.weight,
                    batch_size=bs,
                    num_latent_tokens=self.num_latent_tokens,
                    num_proposal_tokens=self.proposal_size,
                    latent_token_id=self.latent_token_id,
                    latent_eps=first_latent_layer.input_layernorm.variance_epsilon,
                    proposal_eps=(
                        first_proposal_layer.input_layernorm.variance_epsilon
                    ),
                )
            )
        else:
            from sglang.kernels.ops.speculative.dspark.latent_init import (
                latent_init_rmsnorm,
            )

            block_embeds = input_embeds.view(bs, self.block_size, -1)
            # Preserve the reference input layout: anchor followed by masks.
            proposal = block_embeds[:, : self.proposal_size]
            latent, latent_residual = latent_init_rmsnorm(
                block_embeds[:, 0],
                self.embed_tokens.weight[self.latent_token_id],
                first_latent_layer.input_layernorm.weight,
                num_latent_tokens=self.num_latent_tokens,
                eps=first_latent_layer.input_layernorm.variance_epsilon,
            )
            proposal_residual = None

        backend = get_attn_backend()
        latent_batch, proposal_batch, latent_cache_locs = self._make_stage_batches(
            forward_batch, latent_pos_flat, proposal_pos_flat
        )

        # The first proposal layer receives a pre-normalized embedding and its
        # QKV/position transform is independent of every latent-layer result.
        # Overlap that low-M projection with the latent stack.  Joining the
        # streams below preserves the exact trained execution order at the
        # first operation that consumes the projected tensors.
        proposal_flat = proposal.reshape(-1, proposal.shape[-1])
        first_latent_qkv = None
        if token_qkv_cached:
            from sglang.kernels.ops.speculative.dspark.latent_init import (
                cached_qkv_rope,
            )

            latent_attn = first_latent_layer.self_attn
            first_latent_qkv_packed = cached_qkv_rope(
                input_ids,
                self._first_latent_token_qkv_cache,
                latent_attn.rotary_emb.cos_sin_cache,
                latent_pos_flat,
                num_q_heads=latent_attn.num_heads,
                num_k_heads=latent_attn.num_kv_heads,
                head_dim=latent_attn.head_dim,
                tokens_per_batch=self.num_latent_tokens,
                inputs_per_batch=self.proposal_size,
                latent_token_id=self.latent_token_id,
            )
            first_latent_qkv = first_latent_qkv_packed.split(
                [latent_attn.q_size, latent_attn.kv_size, latent_attn.kv_size],
                dim=-1,
            )
        first_proposal_qkv = None
        current_stream = None
        if envs.SGLANG_LATENTSPEC_OVERLAP_PROPOSAL_QKV.get():
            if self._proposal_projection_stream is None:
                self._proposal_projection_stream = torch.cuda.Stream(
                    device=proposal.device
                )
            current_stream = torch.cuda.current_stream(proposal.device)
            self._proposal_projection_stream.wait_stream(current_stream)
            with torch.cuda.stream(self._proposal_projection_stream):
                if token_qkv_cached:
                    proposal_attn = first_proposal_layer.self_attn
                    proposal_qkv_packed = cached_qkv_rope(
                        input_ids,
                        self._first_proposal_token_qkv_cache,
                        proposal_attn.rotary_emb.cos_sin_cache,
                        proposal_pos_flat,
                        num_q_heads=proposal_attn.num_heads,
                        num_k_heads=proposal_attn.num_kv_heads,
                        head_dim=proposal_attn.head_dim,
                        tokens_per_batch=self.proposal_size,
                        inputs_per_batch=self.proposal_size,
                    )
                    first_proposal_qkv = proposal_qkv_packed.split(
                        [
                            proposal_attn.q_size,
                            proposal_attn.kv_size,
                            proposal_attn.kv_size,
                        ],
                        dim=-1,
                    )
                else:
                    first_proposal_qkv = (
                        first_proposal_layer.self_attn.project_current(
                            proposal_flat, proposal_pos_flat
                        )
                    )
        elif token_qkv_cached:
            proposal_attn = first_proposal_layer.self_attn
            proposal_qkv_packed = cached_qkv_rope(
                input_ids,
                self._first_proposal_token_qkv_cache,
                proposal_attn.rotary_emb.cos_sin_cache,
                proposal_pos_flat,
                num_q_heads=proposal_attn.num_heads,
                num_k_heads=proposal_attn.num_kv_heads,
                head_dim=proposal_attn.head_dim,
                tokens_per_batch=self.proposal_size,
                inputs_per_batch=self.proposal_size,
            )
            first_proposal_qkv = proposal_qkv_packed.split(
                [
                    proposal_attn.q_size,
                    proposal_attn.kv_size,
                    proposal_attn.kv_size,
                ],
                dim=-1,
            )

        # Stage 1: exactly four latent queries per request. All four attend to
        # the committed context and to the complete four-token latent block.
        self._set_stage(backend, "latent")
        for latent_layer_id, layer in enumerate(
            self.layers[: self.num_latent_layers]
        ):
            latent, latent_residual = self._run_layer(
                layer,
                latent,
                latent_residual,
                latent_pos_flat,
                latent_batch,
                input_is_normalized=latent_layer_id == 0,
                projected_current=(
                    first_latent_qkv if latent_layer_id == 0 else None
                ),
            )
        if latent_residual is None:
            latent = self.latent_hidden_norm(latent)
        else:
            latent, _ = self.latent_hidden_norm(latent, latent_residual)

        # Stage 2: materialize the fixed latent K/V for each proposal layer,
        # then run exactly seven proposal queries. The stage metadata treats
        # those four cache slots as prefix, yielding context+latent+proposal KV.
        self._set_stage(backend, "proposal")
        if current_stream is not None:
            current_stream.wait_stream(self._proposal_projection_stream)
        proposal = proposal_flat
        proposal_layers = self.layers[self.num_latent_layers :]
        stacked_latent_kv = None
        if self._stacked_latent_kv_weight is not None:
            stacked_latent_kv = F.linear(
                latent,
                self._stacked_latent_kv_weight,
                self._stacked_latent_kv_bias,
            )
        fused_latent_kv_written = (
            stacked_latent_kv is not None
            and self._write_fused_latent_kv(
                stacked_latent_kv,
                latent_pos_flat,
                latent_cache_locs,
            )
        )
        if stacked_latent_kv is not None:
            stacked_latent_kv = stacked_latent_kv.view(
                latent.shape[0], self.num_proposal_layers, -1
            )
        for proposal_layer_id, layer in enumerate(proposal_layers):
            if not fused_latent_kv_written:
                self._write_latent_kv(
                    layer,
                    latent,
                    latent_pos_flat,
                    latent_cache_locs,
                    backend,
                    (
                        None
                        if stacked_latent_kv is None
                        else stacked_latent_kv[:, proposal_layer_id]
                    ),
                )
            proposal, proposal_residual = self._run_layer(
                layer,
                proposal,
                proposal_residual,
                proposal_pos_flat,
                proposal_batch,
                input_is_normalized=(
                    proposal_is_normalized and proposal_layer_id == 0
                ),
                projected_current=(
                    first_proposal_qkv if proposal_layer_id == 0 else None
                ),
            )

        if proposal_residual is None:
            proposal = self.norm(proposal)
        else:
            proposal, _ = self.norm(proposal, proposal_residual)
        hidden_states = proposal.view(bs, self.proposal_size, -1)
        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states.reshape(-1, hidden_states.shape[-1]),
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        stacked = (
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj_noise", "k"),
            ("qkv_proj", "v_proj_noise", "v"),
            ("qkv_proj", "k_proj_mask", "k"),
            ("qkv_proj", "v_proj_mask", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        )
        packed_kv = (
            ("kv_latent", "k_proj_latent", 0),
            ("kv_latent", "v_proj_latent", 1),
            ("kv_ctx", "k_proj", 0),
            ("kv_ctx", "v_proj", 1),
        )
        for raw_name, value in weights:
            name = raw_name.removeprefix("model.")
            if name.startswith("latent_layers."):
                name = "layers." + name[len("latent_layers.") :]
            elif name.startswith("layers."):
                parts = name.split(".", 2)
                name = f"layers.{int(parts[1]) + self.num_latent_layers}.{parts[2]}"
            loaded = False
            for packed, source, shard in stacked:
                if f".{source}." in name:
                    dst = name.replace(source, packed)
                    if dst in params:
                        loader = getattr(
                            params[dst], "weight_loader", default_weight_loader
                        )
                        loader(params[dst], value, shard)
                        loaded = True
                        break
            if not loaded:
                for packed, source, shard in packed_kv:
                    if f".{source}." not in name:
                        continue
                    dst = name.replace(source, packed)
                    if dst in params:
                        param = params[dst]
                        shard_size = param.shape[0] // 2
                        if value.shape[0] != shard_size:
                            raise ValueError(
                                f"Unexpected {source} shard shape {tuple(value.shape)} "
                                f"for packed parameter {dst} {tuple(param.shape)}."
                            )
                        param.data.narrow(0, shard * shard_size, shard_size).copy_(
                            value
                        )
                        loaded = True
                        break
            if loaded or name not in params:
                continue
            loader = getattr(params[name], "weight_loader", default_weight_loader)
            loader(params[name], value)

        proposal_attn = [
            layer.self_attn
            for layer in self.layers[self.num_latent_layers :]
        ]
        self._stacked_latent_kv_weight = torch.cat(
            [attn.kv_latent.weight.detach() for attn in proposal_attn], dim=0
        )
        if proposal_attn and proposal_attn[0].kv_latent.bias is not None:
            self._stacked_latent_kv_bias = torch.cat(
                [attn.kv_latent.bias.detach() for attn in proposal_attn], dim=0
            )


class Qwen3MySpecModel(DSparkDraftMixin, _LatentSpecBackbone):
    """LatentSpec backbone with the checkpoint's DSpark proposal heads.

    The mixin supplies the exact Markov sampling and confidence interfaces used
    by ``DSparkWorkerV2``.  The backbone keeps the two-stage attention geometry
    and position ids identical to training.
    """

    def __new__(cls, config, quant_config=None, prefix: str = ""):
        # Plain LatentSpec checkpoints share this architecture tag but have no
        # Markov head. Keep them on the original DFlash worker/model contract.
        if int(getattr(config, "markov_rank", 0)) <= 0:
            return _LatentSpecBackbone(
                config=config, quant_config=quant_config, prefix=prefix
            )
        return super().__new__(cls)

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        # The draft KV cache owns both latent and proposal attention stacks.
        self.num_stages = len(self.layers)

    def _build_fused_kv_write_bundle(self, pool):
        layers = list(self.layers)
        if not layers:
            return None
        if not (hasattr(pool, "get_key_buffer") and hasattr(pool, "get_value_buffer")):
            return None
        attn0 = layers[0].self_attn
        head_dim = attn0.head_dim
        kv_size = attn0.kv_size
        rotary = attn0.rotary_emb
        if type(rotary).__name__ != "RotaryEmbedding":
            return None
        if getattr(rotary, "rotary_dim", None) != head_dim:
            return None
        eps = attn0.k_norm.variance_epsilon
        weights, knws, meta_rows = [], [], []
        for layer in layers:
            attn = layer.self_attn
            ctx_proj = attn.kv_ctx
            if ctx_proj.bias is not None:
                return None
            if attn.attn.k_scale is not None or attn.attn.v_scale is not None:
                return None
            if attn.head_dim != head_dim or attn.kv_size != kv_size:
                return None
            if attn.rotary_emb is not rotary and not torch.equal(
                attn.rotary_emb.cos_sin_cache, rotary.cos_sin_cache
            ):
                return None
            if attn.rotary_emb.is_neox_style != rotary.is_neox_style:
                return None
            if attn.k_norm.variance_epsilon != eps:
                return None
            k_buf = pool.get_key_buffer(attn.attn.layer_id)
            v_buf = pool.get_value_buffer(attn.attn.layer_id)
            num_kv_heads = kv_size // head_dim
            for buf in (k_buf, v_buf):
                if buf.dtype != torch.bfloat16:
                    return None
                if buf.shape[1:] != (num_kv_heads, head_dim):
                    return None
                if buf.stride(1) != head_dim or buf.stride(2) != 1:
                    return None
            if ctx_proj.weight.dtype != torch.bfloat16:
                return None
            weights.append(ctx_proj.weight)
            knws.append(attn.k_norm.weight.data)
            meta_rows.append(
                [k_buf.data_ptr(), v_buf.data_ptr(), k_buf.stride(0), v_buf.stride(0)]
            )
        device = weights[0].device
        return (
            torch.cat(weights, dim=0).contiguous(),
            torch.tensor(meta_rows, dtype=torch.int64, device=device),
            torch.stack(knws).to(device),
            rotary.cos_sin_cache.to(device),
            eps,
            len(layers),
            kv_size,
            head_dim,
        )

    def _stacked_ctx_kv_params(self) -> Optional[dict]:
        if not envs.SGLANG_DSPARK_STACKED_CTX_KV.get():
            return None
        cached = getattr(self, "_stacked_ctx_kv_cache", False)
        if cached is not False:
            return cached
        weights, biases, k_norm_weights = [], [], []
        eps = None
        for layer in self.layers:
            attn = layer.self_attn
            ctx_proj = attn.kv_ctx
            if eps not in (None, attn.k_norm.variance_epsilon):
                self._stacked_ctx_kv_cache = None
                return None
            eps = attn.k_norm.variance_epsilon
            weights.append(ctx_proj.weight)
            biases.append(ctx_proj.bias)
            k_norm_weights.append(attn.k_norm.weight)
        has_bias = [bias is not None for bias in biases]
        if any(has_bias) and not all(has_bias):
            self._stacked_ctx_kv_cache = None
            return None
        self._stacked_ctx_kv_cache = {
            "weight": torch.cat(weights, dim=0),
            "bias": torch.cat(biases, dim=0) if all(has_bias) else None,
            "k_norm_weight": torch.stack(k_norm_weights, dim=0).float(),
            "eps": eps,
        }
        return self._stacked_ctx_kv_cache

    def prune_to_ctx_kv_injection(self) -> None:
        self.markov_head = None
        self.confidence_head = None
        for layer in self.layers:
            layer.mlp = None
            layer.self_attn.o_proj = None
        torch.cuda.empty_cache()


EntryClass = [Qwen3MySpecModel]
