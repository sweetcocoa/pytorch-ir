# 확장 가이드

이 문서는 NPU IR 프레임워크에 커스텀 연산자를 추가하는 방법을 설명합니다.

## 1. 개요

프레임워크는 **모든 ATen 연산자를 자동으로 처리**합니다. 커스텀 등록이 필요한 경우는 다음과 같습니다:

- **Non-ATen op**: `torch.ops.aten.*`으로 resolve할 수 없는 연산자
- **특수한 변환 로직**: 기본 변환(`_default_conversion`)과 다른 OpNode 구조가 필요한 경우
- **특수한 실행 로직**: ATen fallback이 처리할 수 없는 실행 방식이 필요한 경우

대부분의 경우 아무 것도 등록하지 않아도 됩니다.

## 2. 연산자 레지스트리 이해

### 2.1 레지스트리 구조

```python
# npu_ir/ops/registry.py

# IR 변환 함수 저장 (커스텀 변환용)
_CONVERSION_REGISTRY: Dict[str, Callable] = {}

# 실행 함수 저장 (커스텀 실행용)
_EXECUTION_REGISTRY: Dict[str, Callable] = {}
```

### 2.2 처리 우선순위

**IR 변환** (`converter.py`):
1. `_CONVERSION_REGISTRY`에 등록된 커스텀 변환 함수 확인
2. 없으면 `_default_conversion()` 사용 (모든 ATen op 자동 처리)

**실행** (`executor.py`):
1. `_EXECUTION_REGISTRY`에 등록된 커스텀 실행 함수 확인
2. 없으면 `_aten_fallback()` 사용 (schema 기반 ATen op 자동 실행)

### 2.3 연산자 이름 패턴

ATen 연산자 이름은 다음 패턴을 따릅니다:

```
aten.<op_name>.<overload>
```

예시:
- `aten.conv2d.default`
- `aten.linear.default`
- `aten.add.Tensor`
- `aten.softmax.int`

## 3. 커스텀 실행 함수 등록

ATen fallback이 처리할 수 없는 non-ATen op에 대해서만 필요합니다.

### 3.1 기본 구조

```python
from npu_ir.ops import register_executor
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
    return [result]  # 항상 리스트로 반환
```

### 3.2 입출력 규칙

- **입력**: `List[torch.Tensor]` - IR 노드의 입력 순서대로
- **출력**: `List[torch.Tensor]` - IR 노드의 출력 순서대로
- 단일 출력이라도 리스트로 반환해야 함

### 3.3 다중 출력 예시

```python
@register_executor("my_op_with_two_outputs")
def execute_my_op(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> List[torch.Tensor]:
    x = inputs[0]
    values, indices = x.topk(attrs.get("k", 1), dim=attrs.get("dim", -1))
    return [values, indices]  # 두 개의 출력
```

## 4. 커스텀 IR 변환 함수 등록 (선택사항)

기본 변환기(`_default_conversion`)가 대부분의 경우 충분하지만, OpNode의 구조를 커스터마이즈하고 싶은 경우:

```python
from npu_ir.ops import register_op
from npu_ir import OpNode
from npu_ir.analyzer import NodeInfo

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

### 4.1 NodeInfo 구조

변환 함수는 `NodeInfo` 객체를 받습니다:

```python
@dataclass
class NodeInfo:
    name: str                       # 노드 이름 (예: "conv2d_1")
    op: str                         # 연산 종류 ("call_function")
    target: Any                     # 연산 대상 (예: torch.ops.aten.conv2d.default)
    args: Tuple[Any, ...]           # FX 노드 인자
    kwargs: Dict[str, Any]          # FX 노드 키워드 인자
    input_metas: List[TensorMeta]   # 입력 텐서 메타데이터
    output_metas: List[TensorMeta]  # 출력 텐서 메타데이터
    attrs: Dict[str, Any]           # 추출된 속성 (schema 기반 자동 추출)
```

### 4.2 속성 자동 추출

`node_info.attrs`에는 OpOverload schema 기반으로 모든 non-Tensor 인자가 자동 추출되어 있습니다. 추가 추출 없이 그대로 사용할 수 있습니다.

## 5. 완전한 예시: Non-ATen 커스텀 op

```python
# my_custom_ops.py
import torch
from typing import List, Dict, Any

from npu_ir.ops import register_op, register_executor
from npu_ir import OpNode
from npu_ir.analyzer import NodeInfo


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

### 사용 방법

```python
# 커스텀 연산자 모듈 임포트 (등록됨)
import my_custom_ops

from npu_ir import extract_ir
```

## 6. 모듈로 구성하기

```
my_project/
├── my_ops/
│   ├── __init__.py          # 모든 하위 모듈 import
│   ├── custom_gate.py       # @register_executor 포함
│   └── custom_pooling.py    # @register_executor 포함
└── main.py
```

```python
# my_ops/__init__.py
from . import custom_gate
from . import custom_pooling
# import 시 자동으로 등록됨
```

## 7. 디버깅 팁

### 7.1 FX 그래프 확인

```python
from npu_ir import export_model

exported = export_model(model, inputs, strict=False)

# FX 그래프 출력
print(exported.graph_module.graph)

# 개별 노드 확인
for node in exported.graph_module.graph.nodes:
    if node.op == "call_function":
        print(f"Node: {node.name}")
        print(f"  Target: {node.target}")
        print(f"  Args: {node.args}")
```

### 7.2 등록 확인

```python
from npu_ir.ops.registry import get_conversion_fn, get_execution_fn

op_type = "my_custom_op"
print(f"Conversion: {get_conversion_fn(op_type)}")
print(f"Execution: {get_execution_fn(op_type)}")
```

### 7.3 ATen op schema 확인

```python
import torch

fn = torch.ops.aten.conv2d.default
for arg in fn._schema.arguments:
    print(f"  {arg.name}: {arg.type} (kwarg_only={arg.kwarg_only})")
```

## 8. 주의사항

### 8.1 ATen op은 등록 불필요

ATen op(`aten.*`)에 대해 `@register_executor`를 등록하면 ATen fallback 대신 커스텀 함수가 호출됩니다. 특별한 이유가 없다면 등록하지 마세요 — fallback이 schema 기반으로 올바르게 처리합니다.

### 8.2 입력 순서

FX 그래프의 입력 순서와 실행 함수의 입력 순서가 일치해야 합니다.

### 8.3 속성 기본값

속성이 없을 경우를 대비해 기본값을 제공하세요:

```python
def execute_my_op(inputs, attrs):
    param = attrs.get("param", 1.0)  # 기본값 제공
```

## 9. 기여 가이드

프레임워크에 기여하려면:

1. Non-ATen op의 실행 함수는 `npu_ir/ops/aten_impl.py`에 추가
2. `tests/`에 테스트 추가
3. `docs/operators.md` 문서 업데이트

ATen op은 자동으로 지원되므로 별도 구현이 필요 없습니다.
