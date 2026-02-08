# Extension Guide

This document explains how to add custom operators to the IR extraction framework.

## 1. Overview

The framework **automatically handles all ATen operators**. Custom registration is only needed in the following cases:

- **Non-ATen op**: Operators that cannot be resolved to `torch.ops.aten.*`
- **Special conversion logic**: When OpNode structure different from the default conversion (`_default_conversion`) is needed
- **Special execution logic**: When execution method that ATen fallback cannot handle is required

In most cases, you don't need to register anything.

## 2. Understanding the Operator Registry

### 2.1 Registry Structure

```python
# torch_ir/ops/registry.py

# Store IR conversion functions (for custom conversion)
_CONVERSION_REGISTRY: Dict[str, Callable] = {}

# Store execution functions (for custom execution)
_EXECUTION_REGISTRY: Dict[str, Callable] = {}
```

### 2.2 Processing Priority

**IR Conversion** (`converter.py`):
1. Check for custom conversion function registered in `_CONVERSION_REGISTRY`
2. If not found, use `_default_conversion()` (automatically handles all ATen ops)

**Execution** (`executor.py`):
1. Check for custom execution function registered in `_EXECUTION_REGISTRY`
2. If not found, use `_aten_fallback()` (schema-based automatic ATen op execution)

### 2.3 Operator Name Patterns

ATen operator names follow this pattern:

```
aten.<op_name>.<overload>
```

Examples:
- `aten.conv2d.default`
- `aten.linear.default`
- `aten.add.Tensor`
- `aten.softmax.int`

## 3. Registering Custom Execution Functions

Only needed for non-ATen ops that ATen fallback cannot handle.

### 3.1 Basic Structure

```python
from torch_ir.ops import register_executor
import torch
from typing import List, Dict, Any

@register_executor("my_custom_op")
def execute_my_custom_op(
    inputs: List[torch.Tensor],
    attrs: Dict[str, Any]
) -> List[torch.Tensor]:
    """Execute my_custom_op."""
    x = inputs[0]
    param = attrs.get("param", 1.0)
    result = some_operation(x, param)
    return [result]  # Always return as list
```

### 3.2 Input/Output Rules

- **Input**: `List[torch.Tensor]` - in the order of IR node inputs
- **Output**: `List[torch.Tensor]` - in the order of IR node outputs
- Must return as list even for single output

### 3.3 Multi-output Example

```python
@register_executor("my_op_with_two_outputs")
def execute_my_op(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> List[torch.Tensor]:
    x = inputs[0]
    values, indices = x.topk(attrs.get("k", 1), dim=attrs.get("dim", -1))
    return [values, indices]  # Two outputs
```

## 4. Registering Custom IR Conversion Functions (Optional)

The default converter (`_default_conversion`) is sufficient in most cases, but if you want to customize the OpNode structure:

```python
from torch_ir.ops import register_op
from torch_ir import OpNode
from torch_ir.analyzer import NodeInfo

@register_op("my_custom_op")
def convert_my_custom_op(node_info: NodeInfo) -> OpNode:
    """Custom conversion with extra processing."""
    return OpNode(
        name=node_info.name,
        op_type="my_custom_op",
        inputs=node_info.input_metas,
        outputs=node_info.output_metas,
        attrs={**node_info.attrs, "extra_info": "custom_value"},
    )
```

### 4.1 NodeInfo Structure

Conversion functions receive a `NodeInfo` object:

```python
@dataclass
class NodeInfo:
    name: str                       # Node name (e.g., "conv2d_1")
    op: str                         # Operation type ("call_function")
    target: Any                     # Operation target (e.g., torch.ops.aten.conv2d.default)
    args: Tuple[Any, ...]           # FX node arguments
    kwargs: Dict[str, Any]          # FX node keyword arguments
    input_metas: List[TensorMeta]   # Input tensor metadata
    output_metas: List[TensorMeta]  # Output tensor metadata
    attrs: Dict[str, Any]           # Extracted attributes (auto-extracted based on schema)
```

### 4.2 Automatic Attribute Extraction

`node_info.attrs` contains all non-Tensor arguments automatically extracted based on the OpOverload schema. You can use it as-is without additional extraction.

## 5. Complete Example: Non-ATen Custom Op

```python
# my_custom_ops.py
import torch
from typing import List, Dict, Any

from torch_ir.ops import register_op, register_executor
from torch_ir import OpNode
from torch_ir.analyzer import NodeInfo


@register_op("custom.fused_gate")
def convert_fused_gate(node_info: NodeInfo) -> OpNode:
    return OpNode(
        name=node_info.name,
        op_type="custom.fused_gate",
        inputs=node_info.input_metas,
        outputs=node_info.output_metas,
        attrs=node_info.attrs,
    )


@register_executor("custom.fused_gate")
def execute_fused_gate(
    inputs: List[torch.Tensor],
    attrs: Dict[str, Any]
) -> List[torch.Tensor]:
    """Fused gating: sigmoid(gate) * value."""
    gate = inputs[0]
    value = inputs[1]
    return [torch.sigmoid(gate) * value]
```

### Usage

```python
# Import custom operator module (registers it)
import my_custom_ops

from torch_ir import extract_ir
```

## 6. Organizing as Module

```
my_project/
├── my_ops/
│   ├── __init__.py          # Import all submodules
│   ├── custom_gate.py       # Contains @register_executor
│   └── custom_pooling.py    # Contains @register_executor
└── main.py
```

```python
# my_ops/__init__.py
from . import custom_gate
from . import custom_pooling
# Automatically registered on import
```

## 7. Debugging Tips

### 7.1 Checking FX Graph

```python
from torch_ir import export_model

exported = export_model(model, inputs, strict=False)

# Print FX graph
print(exported.graph_module.graph)

# Check individual nodes
for node in exported.graph_module.graph.nodes:
    if node.op == "call_function":
        print(f"Node: {node.name}")
        print(f"  Target: {node.target}")
        print(f"  Args: {node.args}")
```

### 7.2 Checking Registration

```python
from torch_ir.ops.registry import get_conversion_fn, get_execution_fn

op_type = "my_custom_op"
print(f"Conversion: {get_conversion_fn(op_type)}")
print(f"Execution: {get_execution_fn(op_type)}")
```

### 7.3 Checking ATen Op Schema

```python
import torch

fn = torch.ops.aten.conv2d.default
for arg in fn._schema.arguments:
    print(f"  {arg.name}: {arg.type} (kwarg_only={arg.kwarg_only})")
```

## 8. Precautions

### 8.1 ATen Ops Don't Need Registration

Registering `@register_executor` for ATen ops (`aten.*`) will call the custom function instead of ATen fallback. Unless you have a specific reason, don't register them — fallback handles them correctly based on schema.

### 8.2 Input Order

The input order in the FX graph must match the input order in the execution function.

### 8.3 Attribute Defaults

Provide default values in case attributes are missing:

```python
def execute_my_op(inputs, attrs):
    param = attrs.get("param", 1.0)  # Provide default value
```

## 9. Contribution Guide

To contribute to the framework:

1. Add execution functions for non-ATen ops to `torch_ir/ops/aten_impl.py`
2. Add tests in `tests/`
3. Update `docs/operators.md` documentation

ATen ops are automatically supported, so no separate implementation is needed.
