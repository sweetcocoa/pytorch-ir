"""Extract text-only static IR for Kimi-K2.5 prefill/decode.

This example uses a local, export-friendly text-only Kimi-K2.5 model fork.
It never downloads or loads model weights.

Outputs:
  - kimi_k25_text_prefill_ir.json
  - kimi_k25_text_decode_ir.json
  - kimi_k25_text_kv_mapping.json
"""

from __future__ import annotations

import argparse
import importlib
import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from kimi_k25_text_local import KimiK25TextForCausalLM

from torch_ir import extract_ir, verify_ir_with_state_dict

MODEL_ID = "moonshotai/Kimi-K2.5"


def prepare_text_config(config):
    """Normalize remote config fields used by the local patched model."""
    config._attn_implementation = "eager"
    if getattr(config, "rope_scaling", None) is None and hasattr(config, "to_dict"):
        rope_scaling = config.to_dict().get("rope_scaling")
        if rope_scaling is not None:
            config.rope_scaling = rope_scaling
    return config


def load_default_config():
    """Load the remote HF text config directly."""
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to load the Kimi remote config. "
            "Install it with `uv run --with transformers ...` or add it to your environment."
        ) from exc

    AutoConfig = transformers.AutoConfig
    remote_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    return prepare_text_config(remote_config.text_config)


def make_tiny_config():
    """Build a CPU-friendly tiny config by overriding the remote text config."""
    config = deepcopy(load_default_config())
    overrides = {
        "vocab_size": 64,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "kv_lora_rank": 16,
        "q_lora_rank": 32,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 8,
        "v_head_dim": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "max_position_embeddings": 64,
    }
    for key, value in overrides.items():
        setattr(config, key, value)
    return prepare_text_config(config)


def load_config(preset: str = "default"):
    """Load one of the example config presets."""
    if preset == "default":
        return load_default_config()
    if preset == "tiny":
        return make_tiny_config()
    raise ValueError(f"Unknown config preset: {preset}")


def build_model(config, *, device: str) -> KimiK25TextForCausalLM:
    """Construct the local Kimi model and cast floating params to the config dtype."""
    with torch.device(device):
        model = KimiK25TextForCausalLM(config)
    return model.to(dtype=config.dtype)


def make_additive_causal_mask(
    *,
    query_positions: torch.Tensor,
    key_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a 4D additive causal mask for eager attention."""
    key_positions = torch.arange(key_length, device=device)
    allowed = key_positions.unsqueeze(0) <= query_positions.reshape(-1, 1)
    min_value = torch.tensor(torch.finfo(torch.float32).min, device=device)
    mask = torch.where(allowed, torch.zeros((), device=device), min_value)
    return mask.unsqueeze(0).unsqueeze(0)


class KimiPrefillWrapper(nn.Module):
    """Wrap the local Kimi text model for prefill IR extraction."""

    def __init__(self, model: KimiK25TextForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        result = [outputs.logits]
        for layer_kv in outputs.past_key_values:
            result.append(layer_kv[0])
            result.append(layer_kv[1])
        return tuple(result)


class _IndexCopyCache:
    """Static KV cache that updates full buffers with out-of-place index_copy."""

    def __init__(self, kv_flat, num_layers: int, seen_tokens: int, max_cache_len: int):
        self.key_cache = [kv_flat[2 * i] for i in range(num_layers)]
        self.value_cache = [kv_flat[2 * i + 1] for i in range(num_layers)]
        self._seen_tokens = seen_tokens
        self._max_cache_len = max_cache_len

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if cache_kwargs is None:
            raise ValueError("cache_kwargs must contain cache_position.")
        cache_position = cache_kwargs["cache_position"]
        self.key_cache[layer_idx] = self.key_cache[layer_idx].index_copy(2, cache_position, key_states)
        self.value_cache[layer_idx] = self.value_cache[layer_idx].index_copy(2, cache_position, value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        del layer_idx
        return self._seen_tokens

    def get_max_cache_shape(self, layer_idx=0):
        del layer_idx
        return self._max_cache_len

    def __getitem__(self, idx):
        return self.key_cache[idx], self.value_cache[idx]

    def __iter__(self):
        for idx in range(len(self.key_cache)):
            yield self.key_cache[idx], self.value_cache[idx]

    def __len__(self):
        return len(self.key_cache)


class KimiDecodeWrapper(nn.Module):
    """Wrap the local Kimi text model for decode IR extraction."""

    def __init__(self, model: KimiK25TextForCausalLM, num_layers: int, seen_tokens: int, max_cache_len: int):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.seen_tokens = seen_tokens
        self.max_cache_len = max_cache_len

    def forward(self, input_ids, attention_mask, position_ids, cache_position, *past_kv_flat):
        cache = _IndexCopyCache(past_kv_flat, self.num_layers, self.seen_tokens, self.max_cache_len)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        result = [outputs.logits]
        for idx in range(self.num_layers):
            result.append(cache.key_cache[idx])
            result.append(cache.value_cache[idx])
        return tuple(result)


def _clean_ir_attrs(ir) -> None:
    for node in ir.nodes:
        for key, value in list(node.attrs.items()):
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                node.attrs[key] = str(value)


def save_kv_mapping(
    prefill_ir,
    decode_ir,
    num_layers: int,
    prefill_seq_len: int,
    max_cache_len: int,
    output_path: Path,
):
    """Save explicit prefill/decode KV mapping."""
    prefill_kv = prefill_ir.graph_outputs[1:]
    fixed_inputs = {"input_ids", "attention_mask", "position_ids", "cache_position"}
    decode_kv_in = [meta for meta in decode_ir.graph_inputs if meta.name not in fixed_inputs]
    decode_kv_out = decode_ir.graph_outputs[1:]

    layers = []
    for idx in range(num_layers):
        layers.append(
            {
                "layer": idx,
                "prefill_key_output": prefill_kv[2 * idx].name,
                "prefill_value_output": prefill_kv[2 * idx + 1].name,
                "decode_key_input": decode_kv_in[2 * idx].name,
                "decode_value_input": decode_kv_in[2 * idx + 1].name,
                "decode_key_output": decode_kv_out[2 * idx].name,
                "decode_value_output": decode_kv_out[2 * idx + 1].name,
            }
        )

    mapping = {
        "num_layers": num_layers,
        "prefill_seq_len": prefill_seq_len,
        "max_cache_len": max_cache_len,
        "layers": layers,
    }
    with output_path.open("w") as f:
        json.dump(mapping, f, indent=2)
    return mapping


def build_prefill_meta_inputs(config, prefill_seq_len: int):
    """Create fixed-shape meta inputs for prefill extraction."""
    positions = torch.arange(prefill_seq_len, device="meta").unsqueeze(0)
    attention_mask = make_additive_causal_mask(
        query_positions=positions[0],
        key_length=prefill_seq_len,
        device=torch.device("meta"),
    )
    return (
        torch.randint(0, config.vocab_size, (1, prefill_seq_len), device="meta"),
        attention_mask,
        positions,
    )


def build_decode_meta_inputs(config, prefill_seq_len: int, max_cache_len: int):
    """Create fixed-shape meta inputs for decode extraction."""
    positions = torch.tensor([[prefill_seq_len]], device="meta")
    attention_mask = make_additive_causal_mask(
        query_positions=positions[0],
        key_length=max_cache_len,
        device=torch.device("meta"),
    )
    past_kv_args = []
    head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    for _ in range(config.num_hidden_layers):
        past_kv_args.append(
            torch.randn(1, config.num_attention_heads, max_cache_len, head_dim, device="meta", dtype=config.dtype)
        )
        past_kv_args.append(
            torch.randn(
                1,
                config.num_attention_heads,
                max_cache_len,
                config.v_head_dim,
                device="meta",
                dtype=config.dtype,
            )
        )
    return (
        torch.randint(0, config.vocab_size, (1, 1), device="meta"),
        attention_mask,
        positions,
        torch.tensor([prefill_seq_len], device="meta"),
        *past_kv_args,
    )


def verify_tiny_ir(config, prefill_ir, decode_ir, prefill_seq_len: int, max_cache_len: int) -> None:
    """Run CPU verification for tiny configs using the executor."""
    cpu_model = build_model(config, device="cpu")
    cpu_model.eval()

    print("\n--- Verification (CPU) ---")
    prefill_wrapper = KimiPrefillWrapper(cpu_model)
    prefill_input_ids = torch.randint(0, config.vocab_size, (1, prefill_seq_len))
    prefill_positions = torch.arange(prefill_seq_len).unsqueeze(0)
    prefill_attention_mask = make_additive_causal_mask(
        query_positions=prefill_positions[0],
        key_length=prefill_seq_len,
        device=torch.device("cpu"),
    )
    prefill_valid, prefill_report = verify_ir_with_state_dict(
        prefill_ir,
        prefill_wrapper.state_dict(),
        prefill_wrapper,
        (prefill_input_ids, prefill_attention_mask, prefill_positions),
        rtol=1e-4,
        atol=1e-4,
    )
    print("  prefill:")
    print(f"    {prefill_report}")
    if not prefill_valid:
        raise SystemExit("Prefill IR verification failed.")

    decode_wrapper = KimiDecodeWrapper(cpu_model, config.num_hidden_layers, prefill_seq_len, max_cache_len)
    decode_input_ids = torch.randint(0, config.vocab_size, (1, 1))
    decode_positions = torch.tensor([[prefill_seq_len]])
    decode_attention_mask = make_additive_causal_mask(
        query_positions=decode_positions[0],
        key_length=max_cache_len,
        device=torch.device("cpu"),
    )
    past_kv_args = []
    head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    for _ in range(config.num_hidden_layers):
        past_kv_args.append(torch.randn(1, config.num_attention_heads, max_cache_len, head_dim, dtype=config.dtype))
        past_kv_args.append(
            torch.randn(1, config.num_attention_heads, max_cache_len, config.v_head_dim, dtype=config.dtype)
        )
    decode_valid, decode_report = verify_ir_with_state_dict(
        decode_ir,
        decode_wrapper.state_dict(),
        decode_wrapper,
        (
            decode_input_ids,
            decode_attention_mask,
            decode_positions,
            torch.tensor([prefill_seq_len]),
            *past_kv_args,
        ),
        rtol=1e-4,
        atol=1e-4,
    )
    print("  decode:")
    print(f"    {decode_report}")
    if not decode_valid:
        raise SystemExit("Decode IR verification failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text-only Kimi-K2.5 static IR")
    parser.add_argument(
        "--config-preset",
        choices=["default", "tiny"],
        default="default",
        help="Model config preset to use",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=128, help="Prefill sequence length")
    parser.add_argument("--max-cache-len", type=int, default=2048, help="Max KV cache length for decode")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory for IR files")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run CPU verification with the extracted IR. Intended for the tiny preset.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config_preset)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

    print("=" * 60)
    print("Kimi-K2.5 Text-Only Static IR Extraction")
    print("=" * 60)
    print(f"  num_layers: {num_layers}")
    print(f"  num_attention_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  param_dtype: {config.dtype}")
    print(f"  config_preset: {args.config_preset}")
    print(f"  prefill_seq_len: {args.prefill_seq_len}")
    print(f"  max_cache_len: {args.max_cache_len}")

    model = build_model(config, device="meta")
    model.eval()

    print("\n--- Prefill IR Extraction ---")
    prefill_wrapper = KimiPrefillWrapper(model)
    prefill_inputs = build_prefill_meta_inputs(config, args.prefill_seq_len)
    prefill_ir = extract_ir(prefill_wrapper, prefill_inputs, model_name="KimiK25_Text_Prefill")
    print(f"  nodes: {len(prefill_ir.nodes)}")
    print(f"  weights: {len(prefill_ir.weights)}")

    print("\n--- Decode IR Extraction ---")
    decode_wrapper = KimiDecodeWrapper(model, num_layers, args.prefill_seq_len, args.max_cache_len)
    decode_inputs = build_decode_meta_inputs(config, args.prefill_seq_len, args.max_cache_len)
    decode_ir = extract_ir(decode_wrapper, decode_inputs, model_name="KimiK25_Text_Decode")
    print(f"  nodes: {len(decode_ir.nodes)}")
    print(f"  weights: {len(decode_ir.weights)}")

    if args.verify:
        verify_tiny_ir(config, prefill_ir, decode_ir, args.prefill_seq_len, args.max_cache_len)

    _clean_ir_attrs(prefill_ir)
    _clean_ir_attrs(decode_ir)
    prefill_path = output_dir / "kimi_k25_text_prefill_ir.json"
    prefill_ir.save(prefill_path)
    print(f"  saved: {prefill_path}")

    decode_path = output_dir / "kimi_k25_text_decode_ir.json"
    decode_ir.save(decode_path)
    print(f"  saved: {decode_path}")

    print("\n--- KV Mapping ---")
    mapping_path = output_dir / "kimi_k25_text_kv_mapping.json"
    mapping = save_kv_mapping(prefill_ir, decode_ir, num_layers, args.prefill_seq_len, args.max_cache_len, mapping_path)
    print(f"  saved: {mapping_path} ({len(mapping['layers'])} layers)")

    print("\nDone!")


if __name__ == "__main__":
    main()
