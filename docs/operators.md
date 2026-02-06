# 연산자 지원

이 문서는 NPU IR 프레임워크의 연산자 처리 방식을 설명합니다.

## 1. 개요

프레임워크는 **모든 ATen 연산자를 자동으로 지원**합니다.

- **IR 변환**: `_default_conversion()`이 모든 FX 노드를 `OpNode`로 변환
- **실행**: `_aten_fallback()`이 PyTorch의 op schema를 참조하여 `torch.ops.aten.*`을 직접 호출

개별 연산자마다 변환 함수나 실행 함수를 구현할 필요가 없습니다. PyTorch가 지원하는 모든 ATen op이 자동으로 동작합니다.

## 2. ATen Fallback 동작 방식

### 2.1 Schema 기반 인자 재구성

ATen fallback은 다음 과정으로 op을 실행합니다:

1. `op_type` 문자열(예: `aten.conv2d.default`)에서 `torch.ops.aten.conv2d.default` 함수를 resolve
2. 함수의 `_schema`를 참조하여 인자 타입 정보를 가져옴
3. IR의 flat tensor 입력 목록과 `attrs`를 schema에 맞게 positional/keyword 인자로 재구성
4. 함수를 호출하고 결과를 정규화

### 2.2 지원하는 인자 타입

| Schema 타입 | 처리 방식 |
|------------|----------|
| `Tensor` | 입력 텐서 목록에서 순서대로 할당. 부족하면 attrs에서 스칼라 값 대체 |
| `Tensor[]`, `List[Tensor]` | `_tensor_list_sizes`로 정확한 그룹 크기 결정 |
| `Tensor?[]`, `List[Optional[Tensor]]` | `_tensor_list_none_masks`로 None 위치 복원 |
| `Tensor?`, `Optional[Tensor]` | 입력 텐서가 있으면 할당, 없으면 attrs 확인 후 None |
| 기타 (int, float, bool 등) | `attrs`에서 이름으로 조회 |
| kwarg_only | `kwargs` dict로 전달 (positional이 아님) |

### 2.3 특수 처리

- **Scalar binary ops**: `x * 0.5` 같은 경우, schema상 `Tensor other`이지만 실제로는 스칼라. 텐서 입력이 부족하면 attrs에서 값을 가져옴
- **Device 치환**: attrs의 `device: meta`를 `device: cpu`로 자동 변환 (텐서 생성 op용)
- **Tensor?[] None 복원**: `aten.index.Tensor`의 `[None, idx_tensor]` 같은 패턴에서 None 위치를 정확히 복원

## 3. 검증된 연산자 카테고리

다음 카테고리의 연산자들이 attacker agent 테스트(120+ 모델)를 통해 검증되었습니다:

| 카테고리 | 예시 |
|---------|------|
| Convolution | `conv1d`, `conv2d`, `conv3d`, `conv_transpose2d`, `convolution` |
| Linear/Matrix | `linear`, `addmm`, `mm`, `bmm`, `matmul`, `einsum` |
| Activation | `relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `leaky_relu`, `hardswish` |
| Normalization | `batch_norm`, `layer_norm`, `group_norm`, `instance_norm` |
| Pooling | `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`, `adaptive_max_pool2d` |
| Elementwise | `add`, `sub`, `mul`, `div`, `pow`, `sqrt`, `rsqrt`, `neg`, `abs`, `exp`, `log`, `clamp` |
| Shape | `view`, `reshape`, `permute`, `transpose`, `flatten`, `squeeze`, `unsqueeze`, `expand`, `repeat` |
| Concat/Split | `cat`, `split`, `chunk`, `stack` |
| Reduction | `mean`, `sum`, `max`, `min`, `amax`, `amin` |
| Softmax/Attention | `softmax`, `log_softmax`, `scaled_dot_product_attention` |
| Embedding | `embedding` |
| Indexing | `select`, `slice`, `index`, `gather` |
| Comparison | `eq`, `ne`, `lt`, `gt`, `where` |
| Type/Memory | `to.dtype`, `_to_copy`, `contiguous`, `clone` |
| RNN | `gru`, `lstm` |
| 기타 | `dropout`, `sort`, `topk`, `lerp`, `addcmul` |

## 4. Non-ATen 연산자

ATen fallback이 처리할 수 없는 연산자는 커스텀 실행 함수가 필요합니다.

현재 등록된 non-ATen 실행 함수:

| 연산자 | 설명 |
|--------|------|
| `<built-in function getitem>` | 다중 출력 op에서 특정 출력을 선택 (예: `max(dim=1)` → values, indices) |

## 5. 커스텀 연산자 추가

ATen에 없는 연산자를 사용하는 경우에만 수동 등록이 필요합니다:

```python
from npu_ir.ops import register_executor

@register_executor("my_custom_op")
def execute_my_op(inputs, attrs):
    result = my_custom_operation(inputs[0], **attrs)
    return [result]
```

ATen 연산자는 등록 없이 자동으로 동작합니다. 자세한 내용은 [확장 가이드](extending.md)를 참조하세요.

## 6. 등록된 연산자 확인

```python
from npu_ir import list_registered_ops

ops = list_registered_ops()
print("Custom conversion ops:", len(ops['conversion']))  # 사용자 등록 수
print("Custom execution ops:", len(ops['execution']))    # getitem + 사용자 등록 수
```

## 7. 알려진 제한사항

- **Mixed precision**: `x.half()` 후 float32 weight를 사용하는 경우, ATen op이 자동 캐스팅하지 않아 dtype mismatch 발생 가능
- **Dynamic shapes**: `SymInt` 차원은 `convert_exported_program()` 단계에서 차단됨
- **Meta device 상수**: `forward()`에서 `torch.tensor(...)` 생성 시 meta device에서는 데이터가 없어 `ConversionError` 발생. `self.register_buffer()` 사용 권장
