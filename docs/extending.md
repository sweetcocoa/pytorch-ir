# 확장 가이드

이 문서는 NPU IR 프레임워크에 커스텀 연산자를 추가하는 방법을 설명합니다.

## 1. 개요

프레임워크는 확장 가능한 연산자 레지스트리 시스템을 제공합니다. 새로운 연산자를 지원하려면 두 가지를 구현해야 합니다:

1. **IR 변환 함수**: FX 노드를 OpNode로 변환
2. **실행 함수**: 실제 텐서 연산 수행 (검증에 필요)

## 2. 연산자 레지스트리 이해

### 2.1 레지스트리 구조

```python
# npu_ir/ops/registry.py

# IR 변환 함수 저장
_CONVERSION_REGISTRY: Dict[str, Callable] = {}

# 실행 함수 저장
_EXECUTION_REGISTRY: Dict[str, Callable] = {}
```

### 2.2 연산자 이름 패턴

ATen 연산자 이름은 다음 패턴을 따릅니다:

```
aten.<op_name>.<overload>
```

예시:
- `aten.conv2d.default`
- `aten.linear.default`
- `aten.add.Tensor`
- `aten.softmax.int`

## 3. IR 변환 함수 구현

### 3.1 기본 구조

```python
from npu_ir.ops import register_op
from npu_ir import OpNode, TensorMeta
from npu_ir.analyzer import NodeInfo

@register_op("aten.my_custom_op.default")
def convert_my_custom_op(node_info: NodeInfo) -> OpNode:
    """Convert my_custom_op to OpNode."""
    return OpNode(
        name=node_info.name,
        op_type="aten.my_custom_op.default",
        inputs=node_info.input_metas,
        outputs=node_info.output_metas,
        attrs=node_info.attrs,
    )
```

### 3.2 NodeInfo 구조

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
    attrs: Dict[str, Any]           # 추출된 속성
```

### 3.3 속성 자동 추출

`node_info.attrs`에는 OpOverload schema 기반으로 모든 non-Tensor 인자가 자동 추출되어 있습니다. 대부분의 경우 추가 추출 없이 그대로 사용할 수 있습니다:

```python
@register_op("aten.my_pool.default")
def convert_my_pool(node_info: NodeInfo) -> OpNode:
    # node_info.attrs에 schema 기반 속성이 이미 포함됨
    # 예: {"kernel_size": 3, "stride": 1, "padding": 0, ...}
    return OpNode(
        name=node_info.name,
        op_type="aten.my_pool.default",
        inputs=node_info.input_metas,
        outputs=node_info.output_metas,
        attrs=node_info.attrs,
    )
```

### 3.4 여러 오버로드 등록

같은 함수로 여러 오버로드를 처리할 수 있습니다:

```python
@register_op("aten.my_op.default")
@register_op("aten.my_op.Tensor")
@register_op("aten.my_op.Scalar")
def convert_my_op(node_info: NodeInfo) -> OpNode:
    # 공통 변환 로직
    return OpNode(...)
```

## 4. 실행 함수 구현

### 4.1 기본 구조

```python
from npu_ir.ops import register_executor
import torch
from typing import List, Dict, Any

@register_executor("aten.my_custom_op.default")
def execute_my_custom_op(
    inputs: List[torch.Tensor],
    attrs: Dict[str, Any]
) -> List[torch.Tensor]:
    """Execute my_custom_op."""
    # 입력 텐서 가져오기
    x = inputs[0]

    # 속성 가져오기
    param = attrs.get("param", default_value)

    # 연산 수행
    result = some_operation(x, param)

    # 결과를 리스트로 반환
    return [result]
```

### 4.2 입출력 규칙

- **입력**: `List[torch.Tensor]` - IR 노드의 입력 순서대로
- **출력**: `List[torch.Tensor]` - IR 노드의 출력 순서대로
- 단일 출력이라도 리스트로 반환해야 함

### 4.3 실제 예시: 커스텀 활성화 함수

```python
import torch.nn.functional as F

@register_op("aten.mish.default")
def convert_mish(node_info: NodeInfo) -> OpNode:
    return OpNode(
        name=node_info.name,
        op_type="aten.mish.default",
        inputs=node_info.input_metas,
        outputs=node_info.output_metas,
        attrs={},
    )

@register_executor("aten.mish.default")
def execute_mish(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> List[torch.Tensor]:
    x = inputs[0]
    # Mish(x) = x * tanh(softplus(x))
    result = x * torch.tanh(F.softplus(x))
    return [result]
```

### 4.4 다중 출력 예시

```python
@register_executor("aten.topk.default")
def execute_topk(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> List[torch.Tensor]:
    x = inputs[0]
    k = attrs.get("k", 1)
    dim = attrs.get("dim", -1)

    values, indices = torch.topk(x, k, dim=dim)
    return [values, indices]  # 두 개의 출력
```

## 5. 완전한 예시

### 5.1 커스텀 어텐션 연산

```python
# my_custom_ops.py
import torch
import torch.nn.functional as F
from typing import List, Dict, Any

from npu_ir.ops import register_op, register_executor
from npu_ir import OpNode, TensorMeta
from npu_ir.analyzer import NodeInfo


@register_op("aten.my_attention.default")
def convert_my_attention(node_info: NodeInfo) -> OpNode:
    """Convert custom attention to OpNode."""
    # node_info.attrs에 schema 기반으로 num_heads, dropout_p 등이 자동 포함됨
    return OpNode(
        name=node_info.name,
        op_type="aten.my_attention.default",
        inputs=node_info.input_metas,
        outputs=node_info.output_metas,
        attrs=node_info.attrs,
    )


@register_executor("aten.my_attention.default")
def execute_my_attention(
    inputs: List[torch.Tensor],
    attrs: Dict[str, Any]
) -> List[torch.Tensor]:
    """Execute custom attention."""
    query = inputs[0]
    key = inputs[1]
    value = inputs[2]

    num_heads = attrs.get("num_heads", 8)
    dropout_p = attrs.get("dropout_p", 0.0)

    # Multi-head attention 구현
    batch_size, seq_len, d_model = query.shape
    d_head = d_model // num_heads

    # Reshape for multi-head
    q = query.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    k = key.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    v = value.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)

    # Scaled dot-product attention
    result = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

    # Reshape back
    result = result.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

    return [result]
```

### 5.2 사용 방법

```python
# 커스텀 연산자 모듈 임포트 (등록됨)
import my_custom_ops

import torch
from npu_ir import extract_ir, verify_ir_with_state_dict

# 커스텀 연산을 사용하는 모델
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # ... 모델 정의

    def forward(self, x):
        # torch.ops.aten.my_attention.default 호출
        # (실제로는 torch custom op으로 등록해야 함)
        return x

# IR 추출 및 검증
with torch.device('meta'):
    model = MyModel()
model.eval()

ir = extract_ir(model, inputs)
```

## 6. 모듈로 구성하기

### 6.1 디렉토리 구조

```
my_project/
├── my_ops/
│   ├── __init__.py
│   ├── custom_attention.py
│   ├── custom_activation.py
│   └── custom_pooling.py
└── main.py
```

### 6.2 __init__.py

```python
# my_ops/__init__.py
from . import custom_attention
from . import custom_activation
from . import custom_pooling

# 이 모듈을 import하면 모든 연산자가 등록됨
```

### 6.3 사용

```python
# main.py
import my_ops  # 커스텀 연산자 등록

from npu_ir import extract_ir, list_registered_ops

# 등록된 연산자 확인
ops = list_registered_ops()
print("Custom ops registered:", [op for op in ops['conversion'] if 'my_' in op])

# IR 추출
ir = extract_ir(model, inputs)
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
        print(f"  Kwargs: {node.kwargs}")
```

### 7.2 등록 확인

```python
from npu_ir.ops.registry import get_conversion_fn, get_execution_fn

# 특정 연산자의 등록 여부 확인
op_type = "aten.my_op.default"
print(f"Conversion: {get_conversion_fn(op_type)}")
print(f"Execution: {get_execution_fn(op_type)}")
```

### 7.3 실행 테스트

```python
# 실행 함수 직접 테스트
from my_ops.custom_attention import execute_my_attention

test_inputs = [
    torch.randn(1, 10, 512),  # query
    torch.randn(1, 10, 512),  # key
    torch.randn(1, 10, 512),  # value
]
test_attrs = {"num_heads": 8, "dropout_p": 0.0}

outputs = execute_my_attention(test_inputs, test_attrs)
print(f"Output shape: {outputs[0].shape}")
```

## 8. 주의사항

### 8.1 입력 순서

FX 그래프의 입력 순서와 실행 함수의 입력 순서가 일치해야 합니다.

```python
# FX에서: my_op(x, weight, bias)
# 실행 함수에서:
def execute_my_op(inputs, attrs):
    x = inputs[0]       # 첫 번째 입력
    weight = inputs[1]  # 두 번째 입력
    bias = inputs[2]    # 세 번째 입력
```

### 8.2 속성 기본값

속성이 없을 경우를 대비해 기본값을 제공하세요:

```python
def execute_my_op(inputs, attrs):
    param = attrs.get("param", 1.0)  # 기본값 제공
```

### 8.3 텐서 타입

실행 함수는 PyTorch 텐서를 받고 반환합니다:

```python
def execute_my_op(inputs, attrs):
    assert all(isinstance(t, torch.Tensor) for t in inputs)
    result = ...
    assert isinstance(result, torch.Tensor)
    return [result]
```

## 9. 기여 가이드

프레임워크에 새 연산자를 기여하려면:

1. `npu_ir/ops/aten_ops.py`에 변환 함수 추가
2. `npu_ir/ops/aten_impl.py`에 실행 함수 추가
3. `tests/`에 테스트 추가
4. `docs/operators.md` 문서 업데이트
