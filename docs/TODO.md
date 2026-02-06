# NPU IR Framework Implementation Roadmap

## Phase 1: Basic Framework

### 1.1 Project Setup
- [x] uv project initialization
- [x] pyproject.toml configuration
- [x] Directory structure creation

### 1.2 IR Data Structures (`npu_ir/ir.py`)
- [x] TensorMeta dataclass definition
- [x] OpNode dataclass definition
- [x] NPU_IR dataclass definition
- [x] JSON serialization methods

### 1.3 Model Exporter (`npu_ir/exporter.py`)
- [x] Meta device validation function
- [x] torch.export.export() wrapper
- [x] Error handling and guidance messages

### 1.4 Graph Analyzer (`npu_ir/analyzer.py`)
- [x] ExportedProgram graph traversal
- [x] node.meta['val'] shape/dtype extraction
- [x] graph_signature input/output info extraction
- [x] Weight metadata collection

---

## Phase 2: IR Conversion

### 2.1 IR Builder (`npu_ir/converter.py`)
- [x] FX node → OpNode conversion logic
- [x] Input/output TensorMeta generation
- [x] attrs extraction (kernel_size, stride, etc.)

### 2.2 Operator Handling (`npu_ir/ops/`)
- [x] registry.py: Custom operator registration mechanism
- [x] aten_ops.py: Op type string normalization utilities
- [x] All ATen ops auto-handled by `_default_conversion()` + `_aten_fallback()`

### 2.3 Serializer (`npu_ir/serializer.py`)
- [x] NPU_IR → JSON conversion
- [x] JSON → NPU_IR reverse conversion
- [x] File save/load
- [x] Validation

---

## Phase 3: IR Execution and Verification

### 3.1 Weight Loader (`npu_ir/weight_loader.py`)
- [x] .pt file loading
- [x] .safetensors file loading (optional)
- [x] IR.weights matching validation

### 3.2 ATen Operator Execution (`npu_ir/executor.py`)
- [x] Schema-based ATen fallback (`_aten_fallback`) — auto-executes all ATen ops
- [x] Handles Tensor[], Tensor?[], Optional[Tensor], kwarg_only, scalar args
- [x] Device meta→cpu sanitization for tensor creation ops
- [x] `_tensor_list_sizes` / `_tensor_list_none_masks` for correct list reconstruction
- [x] Non-ATen op executor: getitem only (`npu_ir/ops/aten_impl.py`)

### 3.3 IR Executor (`npu_ir/executor.py`)
- [x] IR graph traversal
- [x] Tensor registry (name → tensor) + producer-based output_map
- [x] Per-node execution and result storage
- [x] Final output return

### 3.4 Verifier (`npu_ir/verifier.py`)
- [x] Original model execution
- [x] IR + weight execution
- [x] torch.allclose() comparison
- [x] Error report generation

---

## Phase 4: Extension and Completion

### 4.1 Custom Op Extension
- [x] @register_op / @register_executor decorator for custom non-ATen ops
- [x] ATen fallback handles all standard ops without registration

### 4.2 Error Handling
- [x] Clear error messages
- [x] Dynamic shape detection and blocking (`_validate_static_shapes`)
- [x] Meta-device lifted constant detection

### 4.3 Tests
- [x] test_exporter.py: export operation tests
- [x] test_analyzer.py: metadata extraction tests
- [x] test_converter.py: conversion accuracy tests
- [x] test_executor.py: IR execution tests
- [x] test_verifier.py: verification function tests
- [x] test_models.py: real model integration tests
  - [x] MLP
  - [x] ConvNet
  - [x] Residual blocks

### 4.5 Comprehensive Test System
- [x] Test model registry (tests/models/)
  - [x] Multi-input/output models (SiameseEncoder, MultiTaskHead)
  - [x] Skip connection models (DeepResNet, DenseBlock)
  - [x] Shared weight models (RecurrentUnroll, WeightTying)
  - [x] Attention models (SelfAttention, CrossAttention, TransformerBlock)
- [x] Test framework (tests/testing/)
  - [x] Mermaid DAG generation
  - [x] IR statistics collection
  - [x] Markdown report generation
  - [x] Test runner with verification
- [x] pytest integration (conftest.py)
  - [x] --generate-reports option
  - [x] --output option
  - [x] --category filter
- [x] CLI runner (tests/__main__.py)
  - [x] --list-models
  - [x] --list-categories
  - [x] --model filter
  - [x] --category filter

### 4.4 Examples and Documentation
- [x] examples/basic_usage.py
- [x] examples/resnet_example.py
- [x] README.md
- [x] docs/concepts.md - 개념 및 아키텍처
- [x] docs/setup.md - 환경 설정
- [x] docs/usage.md - 사용 가이드
- [x] docs/api.md - API 레퍼런스
- [x] docs/operators.md - 연산자 지원 목록
- [x] docs/extending.md - 확장 가이드

---

## Future Enhancements

### Dynamic Shape Support
- [ ] Symbolic shape handling
- [ ] Dynamic batch size support
- [ ] Shape guards from torch.export

### Additional Model Support
- [ ] Transformer/BERT models
- [ ] Vision Transformer (ViT)
- [ ] Detection models (YOLO, etc.)

### Optimization
- [ ] Graph optimization passes
- [ ] Operator fusion detection
- [ ] Memory layout optimization hints

### NPU-Specific Features
- [ ] NPU-specific op mappings
- [ ] Quantization support (INT8, FP16)
- [ ] Memory bandwidth estimation

---

## Notes

### torch.export Considerations
- Uses FakeTensor internally (meta tensor subclass)
- Default ATen level decomposition
- Additional decomposition available via `torch._decomp`

### Unsupported Cases
- Dynamic control flow (data-dependent if/for)
- Some custom autograd functions
- Complex Python operations in forward

### Testing Strategy
1. Unit tests: Each component with pytest
2. Model tests: IR extraction with real models (ResNet, MLP, etc.)
3. Shape validation: Extracted shapes match PyTorch execution results
4. JSON validation: Serialization/deserialization round-trip
5. Numerical validation: IR + weight execution matches original model
