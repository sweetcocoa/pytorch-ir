# Advanced Example

This guide shows how to extract static IR from a huge autoregressive MoE model with `pytorch-ir`, using a text-only Kimi-K2.5 workflow as a concrete example.

## Why a Huge-Model Example Matters

For large language models, the export problem is usually not "can I load the weights?" but "can I make the graph exportable at all?"

With `pytorch-ir`, the model and example inputs live on the `meta` device, so IR extraction does not need real parameter values. This keeps the extraction stage feasible even when the original checkpoint is far too large to fit in local RAM or storage.

That does **not** mean extraction is cheap. For huge models, the expensive parts are still:

- `torch.export` tracing and normalization
- graph analysis and IR conversion
- handling large fixed-shape cache interfaces

In practice, huge-model extraction is mostly about making the forward path export-friendly and giving `extract_ir()` a stable, compiler-oriented callable interface.

## Case Study: Kimi-K2.5 Text-Only IR Extraction

The example flow uses three ideas together:

- load the remote Hugging Face config with `AutoConfig(..., trust_remote_code=True)`
- use a **local patched text-only model** for export
- split the export into separate **prefill** and **decode** graphs

Why patch the model locally instead of exporting the upstream repo model directly?

- Kimi-K2.5 is an MoE model
- the original repository code is not written for `meta + torch.export` extraction
- the text backbone needs export-friendly MoE and cache behavior
- the multimodal stack is unnecessary for text-only IR extraction

So the practical extraction target becomes:

- remote config from Hugging Face
- local text-only modeling code with export-friendly patches

## Extraction Script Architecture

The extraction script is organized around the ABI that downstream runtimes actually need.

### 1. Load the remote text config

```python
from copy import deepcopy
from transformers import AutoConfig

MODEL_ID = "moonshotai/Kimi-K2.5"


def prepare_text_config(config):
    config._attn_implementation = "eager"
    if getattr(config, "rope_scaling", None) is None and hasattr(config, "to_dict"):
        rope_scaling = config.to_dict().get("rope_scaling")
        if rope_scaling is not None:
            config.rope_scaling = rope_scaling
    return config


def load_default_config():
    remote_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    return prepare_text_config(remote_config.text_config)
```

Important details:

- the example uses the remote config object directly
- eager attention is forced up front to keep the exported graph predictable
- config normalization happens before model construction

### 2. Instantiate the model on `meta`

```python
import torch
from kimi_k25_text_local import KimiK25TextForCausalLM


def get_model_dtype(config) -> torch.dtype:
    return getattr(config, "dtype", None) or torch.float32


def build_model(config, *, device: str) -> KimiK25TextForCausalLM:
    with torch.device(device):
        model = KimiK25TextForCausalLM(config)
    return model.to(dtype=get_model_dtype(config))
```

This preserves shape and dtype metadata while avoiding real weight allocation.

### 3. Build fixed-shape prefill inputs

```python
def build_prefill_meta_inputs(config, prefill_seq_len: int):
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
```

Prefill is exported with a fixed `(batch=1, seq_len=prefill_seq_len)` interface.

### 4. Build fixed-shape decode inputs

```python
def build_decode_meta_inputs(config, prefill_seq_len: int, max_cache_len: int):
    positions = torch.tensor([[prefill_seq_len]], device="meta")
    attention_mask = make_additive_causal_mask(
        query_positions=positions[0],
        key_length=max_cache_len,
        device=torch.device("meta"),
    )

    past_kv_args = []
    head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    model_dtype = get_model_dtype(config)
    for _ in range(config.num_hidden_layers):
        past_kv_args.append(
            torch.randn(
                1,
                config.num_attention_heads,
                max_cache_len,
                head_dim,
                device="meta",
                dtype=model_dtype,
            )
        )
        past_kv_args.append(
            torch.randn(
                1,
                config.num_attention_heads,
                max_cache_len,
                config.v_head_dim,
                device="meta",
                dtype=model_dtype,
            )
        )

    return (
        torch.randint(0, config.vocab_size, (1, 1), device="meta"),
        attention_mask,
        positions,
        torch.tensor([prefill_seq_len], device="meta"),
        *past_kv_args,
    )
```

Decode is exported with a fixed `(batch=1, seq_len=1, kv_len=max_cache_len)` interface.

## Why Wrappers Are Necessary

`extract_ir()` traces the exact callable you provide. For huge autoregressive models, the raw model callable is usually not the ABI you want to hand to a compiler or runtime.

The wrapper layer solves that mismatch.

### Prefill wrapper

```python
class KimiPrefillWrapper(nn.Module):
    def __init__(self, model):
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
```

Why this wrapper exists:

- the raw model returns structured `past_key_values`
- compiler backends usually want explicit graph outputs
- flattening KV outputs gives a stable output ordering for all layers

### Decode wrapper

```python
class KimiDecodeWrapper(nn.Module):
    def __init__(self, model, num_layers: int, seen_tokens: int, max_cache_len: int):
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
```

Why this wrapper exists:

- decode must accept cache tensors as explicit graph inputs
- decode must return updated cache tensors as explicit graph outputs
- flattening the cache keeps the exported signature deterministic and backend-friendly

Without wrappers, the exported graph interface is tied too closely to model-internal Python objects.

## Why Decode Needs a Static Cache Wrapper

Decode export needs more than a plain wrapper. It needs a cache object whose updates become visible as tensor ops inside the exported graph.

```python
class _IndexCopyCache:
    def __init__(self, kv_flat, num_layers: int, seen_tokens: int, max_cache_len: int):
        self.key_cache = [kv_flat[2 * i] for i in range(num_layers)]
        self.value_cache = [kv_flat[2 * i + 1] for i in range(num_layers)]
        self._seen_tokens = seen_tokens
        self._max_cache_len = max_cache_len

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cache_position = cache_kwargs["cache_position"]
        self.key_cache[layer_idx] = self.key_cache[layer_idx].index_copy(2, cache_position, key_states)
        self.value_cache[layer_idx] = self.value_cache[layer_idx].index_copy(2, cache_position, value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
```

This object exists because decode export needs:

- fixed-size cache buffers
- explicit tensor updates
- no hidden Python-side mutation outside the exported graph interface

Using `index_copy` makes cache updates appear as normal tensor operations, which is exactly what the IR and downstream executors need.

## Final Extraction Calls

The full extraction flow exports two graphs and one mapping file.

```python
with torch.device("meta"):
    model = build_model(config, device="meta")
model.eval()

prefill_wrapper = KimiPrefillWrapper(model)
prefill_inputs = build_prefill_meta_inputs(config, prefill_seq_len)
prefill_ir = extract_ir(prefill_wrapper, prefill_inputs, model_name="KimiK25_Text_Prefill")

decode_wrapper = KimiDecodeWrapper(model, config.num_hidden_layers, prefill_seq_len, max_cache_len)
decode_inputs = build_decode_meta_inputs(config, prefill_seq_len, max_cache_len)
decode_ir = extract_ir(decode_wrapper, decode_inputs, model_name="KimiK25_Text_Decode")
```

After extraction, the example saves:

- prefill IR
- decode IR
- layer-by-layer KV mapping metadata

## Outputs

The example produces three files:

- `kimi_k25_text_prefill_ir.json`
- `kimi_k25_text_decode_ir.json`
- `kimi_k25_text_kv_mapping.json`

Their roles are different:

- `prefill` IR describes the full prompt pass and emits logits plus initial KV tensors
- `decode` IR describes one-token decode with fixed cache inputs and outputs
- `kv_mapping` tells a runtime exactly how prefill KV outputs line up with decode KV inputs and outputs

For large models, this split is often much easier to integrate than trying to force one exported graph to serve both phases.

## What This Example Does Not Do

- It does not download or load actual model weights.
- It does not export the original multimodal Kimi stack.
- It does not document tiny verification mode here.

This page is intentionally focused on the real huge-model extraction path.

## Pitfalls

- The model and example inputs must stay on the `meta` device during extraction.
- Shapes must be fully static. Do not rely on dynamic cache lengths or symbolic sequence dimensions.
- The wrapper signatures define the exported ABI. Changing wrapper inputs or output ordering changes the runtime contract.
- Huge-model extraction time is dominated by export and graph conversion, not by weight loading.
- Remote config compatibility does not automatically make upstream modeling code exportable. Config and model exportability are separate concerns.

## Summary

For huge autoregressive models, successful IR extraction usually depends on three things:

- a remote config that preserves the original architecture
- local patched modeling code that is export-friendly
- wrapper-defined prefill and decode interfaces with explicit static cache tensors

That combination lets `pytorch-ir` extract stable, backend-oriented IR even when the original model is too large to run in a normal weight-loaded setup.
