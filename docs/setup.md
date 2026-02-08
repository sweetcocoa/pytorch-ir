# Environment Setup

This document explains how to configure the development environment for the IR extraction framework.

!!! note
    For basic installation instructions, see [Home](index.md).

## 1. Development Environment Setup

### 1.1 Project Structure

```
my_compiler/
├── torch_ir/              # Main package
│   ├── __init__.py      # Public API
│   ├── ir.py            # IR data structures
│   ├── exporter.py      # torch.export wrapper
│   ├── analyzer.py      # Graph analysis
│   ├── converter.py     # IR conversion
│   ├── serializer.py    # JSON serialization
│   ├── executor.py      # IR executor
│   ├── weight_loader.py # Weight loader
│   ├── verifier.py      # Verifier
│   └── ops/             # Operator registry and utilities
│       ├── __init__.py
│       ├── registry.py  # Custom operator registration mechanism
│       ├── aten_ops.py  # Op type string normalization utilities
│       └── aten_impl.py # Non-ATen op execution (getitem)
├── tests/               # Test code
│   ├── test_exporter.py
│   ├── test_analyzer.py
│   ├── test_converter.py
│   ├── test_executor.py
│   ├── test_verifier.py
│   └── test_models.py
├── examples/            # Example code
│   ├── basic_usage.py
│   └── resnet_example.py
├── docs/                # Documentation
├── pyproject.toml       # Project configuration
└── README.md
```

### 1.2 pyproject.toml

```toml
[project]
name = "torch-ir"
version = "0.1.0"
description = "PyTorch IR extraction framework for compiler backends"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "torchvision>=0.16",
]
safetensors = [
    "safetensors>=0.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
```

### 1.3 IDE Configuration

#### VS Code

`.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.analysis.typeCheckingMode": "basic",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```

#### PyCharm

1. File → Settings → Project → Python Interpreter
2. Select `.venv` virtual environment
3. Run → Edit Configurations → Add pytest

## 2. Troubleshooting

### 2.1 torch.export Errors

**Symptom**: Error when calling `torch.export.export()`

**Solution**:
```python
# Check PyTorch version
import torch
print(torch.__version__)  # Must be 2.1.0 or higher

# Verify torch.export is available
from torch.export import export
```

### 2.2 Meta Device Errors

**Symptom**: `RuntimeError: meta tensors are not supported`

**Solution**:
```python
# Correct meta tensor creation
with torch.device('meta'):
    model = MyModel()

# Or
model = MyModel()
model = model.to('meta')
```

### 2.3 Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'torch_ir'`

**Solution**:
```bash
# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### 2.4 NumPy Warnings

**Symptom**: `UserWarning: Failed to initialize NumPy`

**Solution**:
```bash
# Install NumPy (optional, warning only and does not affect functionality)
pip install numpy
```

## 3. GPU Environment (Optional)

GPU is not required for IR extraction as it only uses the meta device. However, using a GPU during verification can improve performance.

### 3.1 CUDA Environment

```bash
# Install PyTorch for your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3.2 GPU Verification

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## 4. Next Steps

After completing environment setup:

1. Read the [Usage Guide](usage.md) to learn basic usage
2. Run examples in the [examples/](../examples/) directory
3. Check detailed API in the [API Reference](api/index.md)
