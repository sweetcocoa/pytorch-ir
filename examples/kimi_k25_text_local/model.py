# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team.
# Copyright 2025-2026 The Moonshot AI Team.
#
# This file is adapted from the Kimi-K2.5 / DeepSeek-V3 text model code to
# provide a text-only, export-friendly variant for weight-free IR extraction.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CausalLMOutputWithPast:
    """Minimal causal LM output container."""

    logits: torch.Tensor
    past_key_values: object | None = None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)

    bsz, heads, seq_len, dim = q.shape
    q = q.view(bsz, heads, seq_len, dim // 2, 2).transpose(4, 3).reshape(bsz, heads, seq_len, dim)

    bsz, heads, seq_len, dim = k.shape
    k = k.view(bsz, heads, seq_len, dim // 2, 2).transpose(4, 3).reshape(bsz, heads, seq_len, dim)

    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_linear_ramp_mask(min_value: float, max_value: float, dim: int) -> torch.Tensor:
    if min_value == max_value:
        max_value += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_value) / (max_value - min_value)
    return torch.clamp(linear_func, 0, 1)


class KimiK25TextYarnRotaryEmbedding(nn.Module):
    """Yarn rotary embedding used by Kimi-K2.5 text attention."""

    def __init__(self, config: Any):
        super().__init__()
        rope_scaling = config.rope_scaling
        if rope_scaling is None or rope_scaling.get("type") != "yarn":
            raise ValueError("This local Kimi text model only supports yarn rope scaling.")

        self.dim = config.qk_rope_head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.scaling_factor = rope_scaling["factor"]
        self.original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
        self.beta_fast = rope_scaling["beta_fast"]
        self.beta_slow = rope_scaling["beta_slow"]
        self.mscale = rope_scaling["mscale"]
        self.mscale_all_dim = rope_scaling["mscale_all_dim"]

        self.register_buffer("inv_freq", torch.empty(0), persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self.max_seq_len_cached = 0

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        dim = self.dim
        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = (freq_inter * (1 - inv_freq_mask)) + (freq_extra * inv_freq_mask)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        mscale = float(
            _yarn_get_mscale(self.scaling_factor, self.mscale)
            / _yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )
        emb = torch.cat((freqs, freqs), dim=-1)

        self.inv_freq = inv_freq
        self.cos_cached = (emb.cos() * mscale).to(dtype)
        self.sin_cached = (emb.sin() * mscale).to(dtype)
        self.max_seq_len_cached = seq_len

    def forward(self, value_states: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_seq_len_cached < seq_len:
            self._set_cos_sin_cache(seq_len, value_states.device, value_states.dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class KimiK25TextRMSNorm(nn.Module):
    """RMSNorm used by the text backbone."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class KimiK25TextMLP(nn.Module):
    """Dense SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class KimiK25TextMoEGate(nn.Module):
    """Expert router that preserves Kimi's top-k selection behavior."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts,)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.e_score_correction_bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        logits = F.linear(hidden_states.to(torch.float32), self.weight.to(torch.float32), None)
        if self.scoring_func != "sigmoid":
            raise NotImplementedError(f"Unsupported scoring function: {self.scoring_func}")

        scores = logits.sigmoid()
        if self.topk_method != "noaux_tc":
            raise NotImplementedError(f"Unsupported top-k method: {self.topk_method}")

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        group_scores = scores_for_choice.view(batch_size * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(batch_size * seq_len, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(batch_size * seq_len, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class KimiK25TextTensorizedMoE(nn.Module):
    """Export-friendly MoE that keeps routing semantics in tensor form."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.n_shared_experts = config.n_shared_experts
        self.experts = nn.ModuleList(
            [KimiK25TextMLP(config.hidden_size, config.moe_intermediate_size) for _ in range(config.n_routed_experts)]
        )
        self.gate = KimiK25TextMoEGate(config)
        self.shared_experts = None
        if config.n_shared_experts is not None:
            self.shared_experts = KimiK25TextMLP(
                config.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
            )

    def _stack_expert_weights(self, attr: str) -> torch.Tensor:
        return torch.stack([getattr(expert, attr).weight for expert in self.experts], dim=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        tokens = hidden_states.reshape(-1, hidden_states.shape[-1])

        gate_weight = self._stack_expert_weights("gate_proj")
        up_weight = self._stack_expert_weights("up_proj")
        down_weight = self._stack_expert_weights("down_proj")

        gate_values = F.silu(torch.einsum("th,eih->tei", tokens, gate_weight))
        up_values = torch.einsum("th,eih->tei", tokens, up_weight)
        hidden_values = gate_values * up_values
        expert_outputs = torch.einsum("tei,ehi->teh", hidden_values, down_weight)

        gather_index = topk_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        selected_outputs = expert_outputs.gather(1, gather_index)
        mixed_output = (selected_outputs * topk_weight.unsqueeze(-1).to(selected_outputs.dtype)).sum(dim=1)
        mixed_output = mixed_output.view(*orig_shape)

        if self.shared_experts is not None:
            mixed_output = mixed_output + self.shared_experts(identity)
        return mixed_output


class KimiK25TextCache:
    """Dynamic cache for prefill extraction."""

    def __init__(
        self,
        key_cache: Optional[list[torch.Tensor]] = None,
        value_cache: Optional[list[torch.Tensor]] = None,
    ) -> None:
        self.key_cache = key_cache or []
        self.value_cache = value_cache or []

    @classmethod
    def from_legacy_cache(cls, legacy_cache: object | None) -> "KimiK25TextCache":
        if legacy_cache is None:
            return cls()
        if not isinstance(legacy_cache, (tuple, list)):
            raise TypeError("legacy_cache must be a tuple or list of (key, value) pairs.")
        typed_cache = cast(
            list[tuple[torch.Tensor, torch.Tensor]] | tuple[tuple[torch.Tensor, torch.Tensor], ...], legacy_cache
        )
        key_cache = [layer[0] for layer in typed_cache]
        value_cache = [layer[1] for layer in typed_cache]
        return cls(key_cache, value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, object]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)  # type: ignore[arg-type]
            self.value_cache.append(None)  # type: ignore[arg-type]
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key_states), dim=2)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value_states), dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        del layer_idx
        return -1

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple((k, v) for k, v in zip(self.key_cache, self.value_cache))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[idx], self.value_cache[idx]

    def __iter__(self):
        for idx in range(len(self.key_cache)):
            yield self.key_cache[idx], self.value_cache[idx]

    def __len__(self) -> int:
        return len(self.key_cache)


def _get_usable_length(past_key_value: object, new_seq_length: int, layer_idx: int = 0) -> int:
    max_length = past_key_value.get_max_cache_shape(layer_idx)
    previous_seq_length = past_key_value.get_seq_length(layer_idx)
    if max_length is not None and max_length > 0 and previous_seq_length + new_seq_length > max_length:
        return max_length - new_seq_length
    return previous_seq_length


class KimiK25TextAttention(nn.Module):
    """Eager attention implementation for the text backbone."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = KimiK25TextRMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = KimiK25TextRMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=config.attention_bias)
        self.rotary_emb = KimiK25TextYarnRotaryEmbedding(config)

        self.softmax_scale = self.q_head_dim ** (-0.5)
        mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
        if mscale_all_dim:
            mscale = _yarn_get_mscale(config.rope_scaling["factor"], mscale_all_dim)
            self.softmax_scale = self.softmax_scale * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: object | None = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, object | None]:
        batch_size, query_len, _ = hidden_states.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(batch_size, query_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(batch_size, query_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(
                batch_size,
                query_len,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            .transpose(1, 2)
        )
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += _get_usable_length(past_key_value, kv_seq_len, self.layer_idx)
        if position_ids.numel() > 0 and position_ids.device.type != "meta":
            rope_seq_len = max(kv_seq_len, int(position_ids.max().item()) + 1)
        else:
            rope_seq_len = kv_seq_len
        cos, sin = self.rotary_emb(value_states, seq_len=rope_seq_len)
        q_pe, k_pe = _apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(batch_size, self.num_heads, query_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(batch_size, self.num_heads, query_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                {"cache_position": cache_position},
            )
        kv_seq_len = key_states.shape[-2]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        attn_weights = nn.functional.softmax(
            attn_weights + attention_mask,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                query_len,
                self.num_heads * self.v_head_dim,
            )
        )
        return self.o_proj(attn_output), past_key_value


class KimiK25TextDecoderLayer(nn.Module):
    """Single decoder layer with eager attention and tensorized MoE."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = KimiK25TextAttention(config, layer_idx)
        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.mlp = KimiK25TextTensorizedMoE(config)
        else:
            self.mlp = KimiK25TextMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = KimiK25TextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = KimiK25TextRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: object | None = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, object | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


class KimiK25TextModel(nn.Module):
    """Text-only Kimi-K2.5 decoder."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [KimiK25TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = KimiK25TextRMSNorm(config.hidden_size, config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: object | None = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, object | None]:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            seq_len = inputs_embeds.shape[1]
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        if attention_mask is None:
            raise ValueError("attention_mask must be provided as a 4D additive causal mask.")
        if attention_mask.ndim != 4:
            raise ValueError("attention_mask must have shape (batch, 1, query_len, key_len).")

        if use_cache:
            if past_key_values is None:
                past_key_values = KimiK25TextCache()
            elif isinstance(past_key_values, (tuple, list)):
                past_key_values = KimiK25TextCache.from_legacy_cache(past_key_values)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states, past_key_values = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class KimiK25TextForCausalLM(nn.Module):
    """Text-only causal LM head used for static IR extraction."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.model = KimiK25TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: object | None = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        hidden_states, next_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        logits = self.lm_head(hidden_states)
        if hasattr(next_cache, "to_legacy_cache"):
            next_cache = next_cache.to_legacy_cache()
        return CausalLMOutputWithPast(logits=logits, past_key_values=next_cache)
