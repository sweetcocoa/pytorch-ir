# API 레퍼런스

IR 추출 프레임워크의 공개 API 문서입니다.

!!! info
    API 문서는 소스 코드 docstring에서 자동 생성됩니다. 전체 API 레퍼런스는 [영문 페이지](../api/index.md)를 참조하세요.

## 모듈 목록

| 모듈 | 설명 |
|------|------|
| [IR 데이터 구조](ir.md) | `TensorMeta`, `OpNode`, `IR` |
| [Exporter](exporter.md) | `export_model`, `ExportError` |
| [Converter](converter.md) | `IRConverter`, `ConversionError`, `convert_exported_program` |
| [Executor](executor.md) | `IRExecutor`, `execute_ir`, `ExecutionError` |
| [Serializer](serializer.md) | `serialize_ir`, `save_ir`, `load_ir`, `IRSerializer` |
| [Verifier](verifier.md) | `verify_ir`, `verify_ir_with_state_dict`, `IRVerifier` |
| [Weight Loader](weight_loader.md) | `load_weights`, `WeightLoader`, `WeightLoadError` |
| [Ops](ops.md) | `register_op`, `register_executor` |
| [Visualize](visualize.md) | `ir_to_mermaid`, `generate_op_distribution_pie` |
