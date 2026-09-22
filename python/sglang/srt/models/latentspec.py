"""SGLang inference model for the Qwen3 MySpec/LatentSpec checkpoint.

The drafter has two distinct stages per proposal: latent tokens attend to the
materialized target-context KV, then mask/proposal tokens attend to context,
the new latent states, and the complete mask block.  Draft execution is eager
in the first implementation; target verification still uses the normal SGLang
attention backend and CUDA graphs.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashAttention, DFlashDecoderLayer
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config


def _linear(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, module.weight, module.bias)


class _LatentSpecAttention(DFlashAttention):
    """DFlash cache plumbing plus the separate MySpec context projections."""

    def __init__(self, config, layer_id: int, *, with_latent_kv: bool):
        super().__init__(config=config, layer_id=layer_id)
        hidden = int(config.hidden_size)
        kv_out = self.total_num_kv_heads * self.head_dim
        bias = bool(getattr(config, "attention_bias", False))
        self.k_ctx = nn.Linear(hidden, kv_out, bias=bias)
        self.v_ctx = nn.Linear(hidden, kv_out, bias=bias)
        self.k_latent = (
            nn.Linear(hidden, kv_out, bias=bias) if with_latent_kv else None
        )
        self.v_latent = (
            nn.Linear(hidden, kv_out, bias=bias) if with_latent_kv else None
        )

    def kv_proj_only(self, hidden_states: torch.Tensor):
        return _linear(self.k_ctx, hidden_states), _linear(self.v_ctx, hidden_states)

    def project_current(self, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view_as(q)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view_as(k)
        return q, k, v

    def project_latent_kv(self, hidden_states: torch.Tensor):
        assert self.k_latent is not None and self.v_latent is not None
        k = _linear(self.k_latent, hidden_states)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view_as(k)
        return k, _linear(self.v_latent, hidden_states)


class _LatentSpecLayer(DFlashDecoderLayer):
    def __init__(self, config, layer_id: int, *, with_latent_kv: bool):
        nn.Module.__init__(self)
        hidden = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.self_attn = _LatentSpecAttention(
            config, layer_id=layer_id, with_latent_kv=with_latent_kv
        )
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)
        # Reuse the SGLang Qwen3-compatible gated MLP and its weight loader.
        from sglang.srt.models.dflash import DFlashMLP

        self.mlp = DFlashMLP(config)
        self.attention_conv = None
        self.mlp_conv = None


class Qwen3MySpecModel(nn.Module):
    """Native loader/executor for ``Qwen3MySpecModel`` checkpoints.

    TP=1 is intentional for the initial eager draft implementation.  It keeps
    the semantics identical to the reference evaluator while target inference
    may still use arbitrary supported parallelism in a later implementation.
    """

    supports_fused_context_kv = False
    candidate_selector = None

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        del quant_config, prefix
        self.config = config
        self.draft_config = parse_dflash_draft_config(draft_hf_config=config)
        self.proposal_size = int(config.block_size)
        self.block_size = self.proposal_size + 1
        self.num_latent_tokens = int(config.num_latent_tokens)
        self.num_latent_layers = int(config.num_latent_layers)
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
            _LatentSpecLayer(config, i, with_latent_kv=False)
            for i in range(self.num_latent_layers)
        ]
        mask = [
            _LatentSpecLayer(config, self.num_latent_layers + i, with_latent_kv=True)
            for i in range(int(config.num_hidden_layers))
        ]
        # DFlashWorker iterates `layers` to materialize context KV for every
        # attention layer. Keep one flat list and remember the stage boundary.
        self.layers = nn.ModuleList(latent + mask)
        self._runtime_token_to_kv_pool = None
        self._runtime_req_to_token_pool = None

    def set_block_size(self, block_size: int) -> None:
        if int(block_size) != self.block_size:
            raise ValueError(
                "LATENTSPEC proposal width is fixed by the trained checkpoint: "
                f"config.block_size={self.block_size}, requested={block_size}."
            )

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_attention_sliding_window_size(self):
        return None

    def set_runtime_pools(self, *, token_to_kv_pool, req_to_token_pool) -> None:
        self._runtime_token_to_kv_pool = token_to_kv_pool
        self._runtime_req_to_token_pool = req_to_token_pool

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(target_hidden))

    def prepare_context_hidden_for_kv(self, layer, ctx_hidden):
        return ctx_hidden

    def _context_kv(self, layer, forward_batch: ForwardBatch, row: int):
        if (
            self._runtime_token_to_kv_pool is None
            or self._runtime_req_to_token_pool is None
        ):
            raise RuntimeError("LATENTSPEC runtime KV pools were not attached by the worker.")
        req = int(forward_batch.req_pool_indices[row].item())
        length = int(forward_batch.seq_lens[row].item())
        loc = self._runtime_req_to_token_pool.req_to_token[req, :length].long()
        k_all, v_all = self._runtime_token_to_kv_pool.get_kv_buffer(
            layer.self_attn.attn.layer_id
        )
        return k_all[loc], v_all[loc]

    @staticmethod
    def _attend(attn, q, k, v):
        # [L,H,D] -> [H,L,D], expanding grouped KV heads when needed.
        q = q.view(q.shape[0], attn.num_heads, attn.head_dim).transpose(0, 1)
        k = k.view(k.shape[0], attn.num_kv_heads, attn.head_dim).transpose(0, 1)
        v = v.view(v.shape[0], attn.num_kv_heads, attn.head_dim).transpose(0, 1)
        if attn.num_heads != attn.num_kv_heads:
            groups = attn.num_heads // attn.num_kv_heads
            k = k.repeat_interleave(groups, dim=0)
            v = v.repeat_interleave(groups, dim=0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return out.transpose(0, 1).reshape(out.shape[1], -1)

    def _run_layer(self, layer, hidden, positions, context_kv, latent=None):
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        q, k_cur, v_cur = layer.self_attn.project_current(hidden)
        q, k_cur = layer.self_attn.rotary_emb(positions, q, k_cur)
        k_parts = [context_kv[0]]
        v_parts = [context_kv[1]]
        if latent is not None:
            k_lat, v_lat = layer.self_attn.project_latent_kv(latent)
            latent_positions = positions[: self.num_latent_tokens]
            k_lat = layer.self_attn.apply_k_rope(latent_positions, k_lat)
            k_parts.append(
                k_lat.view(
                    -1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim
                )
            )
            v_parts.append(
                v_lat.view(
                    -1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim
                )
            )
        k_parts.append(
            k_cur.view(-1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
        )
        v_parts.append(
            v_cur.view(-1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
        )
        attn_out = self._attend(
            layer.self_attn, q, torch.cat(k_parts), torch.cat(v_parts)
        )
        attn_out, _ = layer.self_attn.o_proj(attn_out)
        hidden = residual + attn_out
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        return residual + layer.mlp(hidden)

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
        all_mask = (
            input_embeds if input_embeds is not None else self.embed_tokens(input_ids)
        ).view(bs, self.block_size, -1)
        mask = all_mask[:, : self.proposal_size]
        pos = positions.view(bs, self.block_size)
        anchor = input_ids.view(bs, self.block_size)[:, :1]
        latent_ids = torch.full(
            (bs, self.num_latent_tokens),
            self.latent_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        latent_ids[:, :1] = anchor
        latent = self.embed_tokens(latent_ids)

        outputs = []
        for row in range(bs):
            latent_row = latent[row]
            latent_pos = pos[row, :1] + torch.arange(
                self.num_latent_tokens, device=pos.device, dtype=pos.dtype
            )
            for layer in self.layers[: self.num_latent_layers]:
                latent_row = self._run_layer(
                    layer, latent_row, latent_pos,
                    self._context_kv(layer, forward_batch, row),
                )
            latent_row = self.latent_hidden_norm(latent_row)
            mask_row = mask[row]
            for layer in self.layers[self.num_latent_layers :]:
                mask_row = self._run_layer(
                    layer,
                    mask_row,
                    pos[row, : self.proposal_size],
                    self._context_kv(layer, forward_batch, row),
                    latent=latent_row,
                )
            outputs.append(self.norm(mask_row))
        proposal_hidden = torch.stack(outputs)
        # DFlash samples hidden[:, 1:] because its position zero is the anchor.
        # MySpec predicts from every one of its seven mask positions, so prepend
        # an unused anchor row and expose all proposal rows at indices 1..7.
        anchor_pad = proposal_hidden.new_zeros((bs, 1, proposal_hidden.shape[-1]))
        hidden_states = torch.cat((anchor_pad, proposal_hidden), dim=1)
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
        for raw_name, value in weights:
            name = raw_name.removeprefix("model.")
            if name.startswith("latent_layers."):
                name = "layers." + name[len("latent_layers.") :]
            elif name.startswith("layers."):
                parts = name.split(".", 2)
                name = f"layers.{int(parts[1]) + self.num_latent_layers}.{parts[2]}"
            name = name.replace("self_attn.k_proj_latent", "self_attn.k_latent")
            name = name.replace("self_attn.v_proj_latent", "self_attn.v_latent")
            name = name.replace(".k_proj.", ".k_ctx.")
            name = name.replace(".v_proj.", ".v_ctx.")
            loaded = False
            for packed, source, shard in stacked:
                if f".{source}." in name:
                    dst = name.replace(source, packed)
                    if dst in params:
                        loader = getattr(params[dst], "weight_loader", default_weight_loader)
                        loader(params[dst], value, shard)
                        loaded = True
                        break
            if loaded or name not in params:
                continue
            loader = getattr(params[name], "weight_loader", default_weight_loader)
            loader(params[name], value)


EntryClass = [Qwen3MySpecModel]
