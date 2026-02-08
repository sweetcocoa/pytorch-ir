# API 레퍼런스

NPU IR 프레임워크의 공개 API 문서입니다.

## 메인 진입점

::: npu_ir.extract_ir

## 모듈 목록

| 모듈 | 설명 |
|------|------|
| [IR 데이터 구조](ir.md) | `TensorMeta`, `OpNode`, `NPU_IR` |
| [Exporter](exporter.md) | `export_model`, `ExportError` |
| [Converter](converter.md) | `IRConverter`, `ConversionError`, `convert_exported_program` |
| [Executor](executor.md) | `IRExecutor`, `execute_ir`, `ExecutionError` |
| [Serializer](serializer.md) | `serialize_ir`, `save_ir`, `load_ir`, `IRSerializer` |
| [Verifier](verifier.md) | `verify_ir`, `verify_ir_with_state_dict`, `IRVerifier` |
| [Weight Loader](weight_loader.md) | `load_weights`, `WeightLoader`, `WeightLoadError` |
| [Ops](ops.md) | `register_op`, `register_executor` |
