# Torch IR Extraction Framework

A library for extracting IR (Intermediate Representation) from PyTorch models.

## Key Features

- **Weight-free Extraction**: Extract only graph structure without loading actual weights into memory using meta tensors
- **torch.export Based**: Uses TorchDynamo-based tracing, the official PyTorch recommended approach
- **Complete Metadata**: Automatic extraction of shape and dtype information for all tensors
- **IR Execution and Verification**: Execute extracted IR and verify identical results with original model
- **Extensible Design**: Provides custom operator registration mechanism
- **CLI Tools**: Inspect and visualize IR files from the terminal (`pytorch-ir info`, `pytorch-ir visualize`)

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Basic Usage

```python
import torch
from torch_ir import extract_ir, verify_ir_with_state_dict

# 1. Create model on meta device
with torch.device('meta'):
    model = MyModel()
model.eval()

# 2. Prepare example inputs (meta device)
example_inputs = (torch.randn(1, 3, 224, 224, device='meta'),)

# 3. Extract IR
ir = extract_ir(model, example_inputs)

# 4. Check IR information
print(f"Nodes: {len(ir.nodes)}")
print(f"Weights: {len(ir.weights)}")
for node in ir.nodes[:5]:
    print(f"  {node.op_type}: {[t.shape for t in node.inputs]} -> {[t.shape for t in node.outputs]}")

# 5. Save IR
ir.save("model_ir.json")
```

### IR Verification

```python
# Compare execution results between original model and IR
original_model = MyModel()
original_model.load_state_dict(torch.load('weights.pt'))
original_model.eval()

test_input = torch.randn(1, 3, 224, 224)
is_valid, report = verify_ir_with_state_dict(
    ir=ir,
    state_dict=original_model.state_dict(),
    original_model=original_model,
    test_inputs=(test_input,),
)

print(f"Verification: {'PASSED' if is_valid else 'FAILED'}")
print(report)
```

## Documentation

- [Concepts and Architecture](concepts.md) - Core concepts and design of the framework
- [Usage Guide](usage.md) - Detailed usage and examples
- [Example](example.md) - Simple extraction and visualization example
- [Advanced Example](advanced-example.md) - Huge-model extraction workflow
- [API Reference](api/index.md) - Public API documentation
- [Operator Support](operators.md) - List of supported ATen operators
- [Extension Guide](extending.md) - How to add custom operators
- [CLI Reference](cli.md) - Command-line tools for IR inspection and visualization
