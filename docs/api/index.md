# API Reference

Public API documentation for the NPU IR framework.

## Main Entry Point

::: npu_ir.extract_ir

## Module List

| Module | Description |
|--------|-------------|
| [IR Data Structures](ir.md) | `TensorMeta`, `OpNode`, `NPU_IR` |
| [Exporter](exporter.md) | `export_model`, `ExportError` |
| [Converter](converter.md) | `IRConverter`, `ConversionError`, `convert_exported_program` |
| [Executor](executor.md) | `IRExecutor`, `execute_ir`, `ExecutionError` |
| [Serializer](serializer.md) | `serialize_ir`, `save_ir`, `load_ir`, `IRSerializer` |
| [Verifier](verifier.md) | `verify_ir`, `verify_ir_with_state_dict`, `IRVerifier` |
| [Weight Loader](weight_loader.md) | `load_weights`, `WeightLoader`, `WeightLoadError` |
| [Ops](ops.md) | `register_op`, `register_executor` |
| [Visualize](visualize.md) | `ir_to_mermaid`, `generate_op_distribution_pie` |
