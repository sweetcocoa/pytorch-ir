# Operator Support

This document explains how the NPU IR framework handles operators.

## 1. Overview

The framework **automatically supports all ATen operators**.

- **IR Conversion**: `_default_conversion()` converts all FX nodes to `OpNode`s
- **Execution**: `_aten_fallback()` directly calls `torch.ops.aten.*` by referencing PyTorch's op schema

There is no need to implement conversion or execution functions for individual operators. All ATen ops supported by PyTorch work automatically.

## 2. How ATen Fallback Works

### 2.1 Schema-based Argument Reconstruction

ATen fallback executes ops through the following process:

1. Resolve the `torch.ops.aten.conv2d.default` function from the `op_type` string (e.g., `aten.conv2d.default`)
2. Get argument type information by referencing the function's `_schema`
3. Reconstruct the flat tensor input list and `attrs` from IR into positional/keyword arguments according to the schema
4. Call the function and normalize the result

### 2.2 Supported Argument Types

| Schema Type | Handling |
|------------|----------|
| `Tensor` | Assigned sequentially from input tensor list. If insufficient, substitute with scalar value from attrs |
| `Tensor[]`, `List[Tensor]` | Determine exact group size using `_tensor_list_sizes` |
| `Tensor?[]`, `List[Optional[Tensor]]` | Restore None positions using `_tensor_list_none_masks` |
| `Tensor?`, `Optional[Tensor]` | Assign if input tensor exists, otherwise check attrs and use None |
| Others (int, float, bool, etc.) | Look up by name in `attrs` |
| kwarg_only | Pass as `kwargs` dict (not positional) |

### 2.3 Special Handling

- **Scalar binary ops**: For cases like `x * 0.5`, the schema specifies `Tensor other` but it's actually a scalar. If tensor inputs are insufficient, fetch the value from attrs
- **Device substitution**: Automatically convert `device: meta` in attrs to `device: cpu` (for tensor creation ops)
- **Tensor?[] None restoration**: Accurately restore None positions in patterns like `[None, idx_tensor]` in `aten.index.Tensor`

## 3. Support Coverage

**All ATen operators are automatically supported via schema-based approach.** Any operator registered in `torch.ops.aten.*` can be converted to IR and executed without separate implementation.

This includes all standard PyTorch operator categories: Convolution, Linear, Activation, Normalization, Pooling, Elementwise, Shape transformation, Reduction, Softmax/Attention, Embedding, Indexing, Comparison, RNN, etc.

## 4. Non-ATen Operators

Operators that ATen fallback cannot handle require custom execution functions.

Currently registered non-ATen execution functions:

| Operator | Description |
|--------|------|
| `<built-in function getitem>` | Select specific output from multi-output op (e.g., `max(dim=1)` → values, indices) |

## 5. Adding Custom Operators

Manual registration is only needed when using operators not in ATen:

```python
from npu_ir.ops import register_executor

@register_executor("my_custom_op")
def execute_my_op(inputs, attrs):
    result = my_custom_operation(inputs[0], **attrs)
    return [result]
```

ATen operators work automatically without registration. See the [Extension Guide](extending.md) for details.

## 6. Checking Registered Operators

```python
from npu_ir import list_registered_ops

ops = list_registered_ops()
print("Custom conversion ops:", len(ops['conversion']))  # User-registered count
print("Custom execution ops:", len(ops['execution']))    # getitem + user-registered count
```

## 7. Known Limitations

- **Mixed precision**: When using `x.half()` followed by float32 weights, dtype mismatch may occur as ATen ops don't auto-cast
- **Dynamic shapes**: `SymInt` dimensions are blocked at the `convert_exported_program()` stage
- **Meta device constants**: Creating `torch.tensor(...)` in `forward()` causes `ConversionError` on meta device due to missing data. Use `self.register_buffer()` instead
