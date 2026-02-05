# 연산자 지원

이 문서는 NPU IR 프레임워크에서 지원하는 ATen 연산자 목록을 설명합니다.

## 1. 개요

프레임워크는 60개 이상의 ATen 연산자를 지원합니다. 각 연산자는 다음 두 가지가 구현되어 있습니다:

- **IR 변환 (aten_ops.py)**: FX 노드를 OpNode로 변환
- **실행 (aten_impl.py)**: 실제 텐서 연산 수행

지원되지 않는 연산자도 기본 변환기로 처리되므로 IR 추출은 가능하지만, 실행/검증은 실패할 수 있습니다.

## 2. Convolution 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.conv1d.default` | ✅ | ✅ | 1D 컨볼루션 |
| `aten.conv2d.default` | ✅ | ✅ | 2D 컨볼루션 |
| `aten.conv3d.default` | ✅ | ✅ | 3D 컨볼루션 |
| `aten.convolution.default` | ✅ | ✅ | 일반 컨볼루션 |
| `aten._conv_depthwise2d` | ✅ | - | Depthwise 컨볼루션 |

**속성:**
- `stride`: 스트라이드
- `padding`: 패딩
- `dilation`: 딜레이션
- `groups`: 그룹 수

## 3. Linear/Matrix 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.linear.default` | ✅ | ✅ | 선형 변환 |
| `aten.addmm.default` | ✅ | ✅ | bias + input @ weight |
| `aten.mm.default` | ✅ | ✅ | 행렬 곱셈 |
| `aten.bmm.default` | ✅ | ✅ | 배치 행렬 곱셈 |
| `aten.matmul.default` | ✅ | ✅ | 일반 행렬 곱셈 |

## 4. 활성화 함수

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.relu.default` | ✅ | ✅ | ReLU |
| `aten.relu_` | ✅ | ✅ | In-place ReLU |
| `aten.gelu.default` | ✅ | ✅ | GELU |
| `aten.silu.default` | ✅ | ✅ | SiLU/Swish |
| `aten.sigmoid.default` | ✅ | ✅ | Sigmoid |
| `aten.tanh.default` | ✅ | ✅ | Tanh |
| `aten.leaky_relu.default` | ✅ | ✅ | Leaky ReLU |
| `aten.hardswish.default` | ✅ | ✅ | Hard Swish |
| `aten.hardsigmoid.default` | ✅ | ✅ | Hard Sigmoid |

## 5. 정규화 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.batch_norm.default` | ✅ | ✅ | 배치 정규화 |
| `aten._native_batch_norm_legit_no_training.default` | ✅ | ✅ | 추론용 배치 정규화 |
| `aten.layer_norm.default` | ✅ | ✅ | 레이어 정규화 |
| `aten.native_layer_norm.default` | ✅ | ✅ | 네이티브 레이어 정규화 |
| `aten.group_norm.default` | ✅ | ✅ | 그룹 정규화 |
| `aten.instance_norm.default` | ✅ | - | 인스턴스 정규화 |

**Batch Norm 속성:**
- `training`: 학습 모드 여부
- `momentum`: 모멘텀
- `eps`: 엡실론

**Layer Norm 속성:**
- `normalized_shape`: 정규화 shape
- `eps`: 엡실론

## 6. Pooling 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.max_pool2d.default` | ✅ | ✅ | 2D 최대 풀링 |
| `aten.max_pool2d_with_indices.default` | ✅ | ✅ | 인덱스 포함 최대 풀링 |
| `aten.avg_pool2d.default` | ✅ | ✅ | 2D 평균 풀링 |
| `aten.adaptive_avg_pool2d.default` | ✅ | ✅ | 적응형 평균 풀링 |
| `aten._adaptive_avg_pool2d.default` | ✅ | ✅ | 적응형 평균 풀링 |
| `aten.adaptive_max_pool2d.default` | ✅ | - | 적응형 최대 풀링 |

**속성:**
- `kernel_size`: 커널 크기
- `stride`: 스트라이드
- `padding`: 패딩
- `output_size`: 출력 크기 (adaptive)

## 7. 원소별 연산

### 7.1 산술 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.add.Tensor` | ✅ | ✅ | 덧셈 |
| `aten.add_.Tensor` | ✅ | ✅ | In-place 덧셈 |
| `aten.sub.Tensor` | ✅ | ✅ | 뺄셈 |
| `aten.mul.Tensor` | ✅ | ✅ | 곱셈 |
| `aten.div.Tensor` | ✅ | ✅ | 나눗셈 |
| `aten.pow.Tensor_Scalar` | ✅ | ✅ | 거듭제곱 |

### 7.2 단항 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.sqrt.default` | ✅ | ✅ | 제곱근 |
| `aten.rsqrt.default` | ✅ | ✅ | 역 제곱근 |
| `aten.neg.default` | ✅ | ✅ | 부정 |
| `aten.abs.default` | ✅ | ✅ | 절대값 |
| `aten.exp.default` | ✅ | ✅ | 지수 |
| `aten.log.default` | ✅ | ✅ | 로그 |
| `aten.clamp.default` | ✅ | ✅ | 클램프 |
| `aten.clamp_min.default` | ✅ | - | 최소값 클램프 |
| `aten.clamp_max.default` | ✅ | - | 최대값 클램프 |

## 8. Shape 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.view.default` | ✅ | ✅ | 뷰 변환 |
| `aten._unsafe_view.default` | ✅ | ✅ | 안전하지 않은 뷰 |
| `aten.reshape.default` | ✅ | ✅ | 재형성 |
| `aten.permute.default` | ✅ | ✅ | 차원 순서 변경 |
| `aten.transpose.int` | ✅ | ✅ | 전치 |
| `aten.t.default` | ✅ | ✅ | 2D 전치 |
| `aten.flatten.using_ints` | ✅ | ✅ | 평탄화 |
| `aten.squeeze.dim` | ✅ | ✅ | 차원 제거 |
| `aten.unsqueeze.default` | ✅ | ✅ | 차원 추가 |
| `aten.expand.default` | ✅ | ✅ | 확장 |
| `aten.repeat.default` | ✅ | - | 반복 |

**Flatten 속성:**
- `start_dim`: 시작 차원
- `end_dim`: 끝 차원

## 9. 연결/분할 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.cat.default` | ✅ | ✅ | 연결 |
| `aten.split.Tensor` | ✅ | ✅ | 분할 |
| `aten.split_with_sizes.default` | ✅ | - | 크기 지정 분할 |
| `aten.chunk.default` | ✅ | - | 청크 분할 |
| `aten.stack.default` | ✅ | ✅ | 스택 |

**속성:**
- `dim`: 연결/분할 차원
- `split_size`: 분할 크기

## 10. 축소 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.mean.dim` | ✅ | ✅ | 평균 |
| `aten.sum.dim_IntList` | ✅ | ✅ | 합계 |
| `aten.max.dim` | ✅ | ✅ | 최대값 |
| `aten.amax.default` | ✅ | ✅ | 축 최대값 |
| `aten.min.dim` | ✅ | ✅ | 최소값 |
| `aten.amin.default` | ✅ | ✅ | 축 최소값 |

**속성:**
- `dim`: 축소 차원
- `keepdim`: 차원 유지 여부

## 11. Softmax/Attention 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.softmax.int` | ✅ | ✅ | Softmax |
| `aten._softmax.default` | ✅ | ✅ | 네이티브 Softmax |
| `aten.log_softmax.int` | ✅ | ✅ | Log Softmax |
| `aten._log_softmax.default` | ✅ | ✅ | 네이티브 Log Softmax |
| `aten.scaled_dot_product_attention.default` | ✅ | ✅ | SDPA |
| `aten._scaled_dot_product_flash_attention.default` | ✅ | - | Flash Attention |
| `aten._scaled_dot_product_efficient_attention.default` | ✅ | - | Efficient Attention |

**Softmax 속성:**
- `dim`: Softmax 차원

**SDPA 속성:**
- `dropout_p`: 드롭아웃 확률
- `is_causal`: 인과적 마스킹
- `scale`: 스케일 팩터

## 12. Embedding 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.embedding.default` | ✅ | ✅ | 임베딩 룩업 |

**속성:**
- `padding_idx`: 패딩 인덱스

## 13. Dropout 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.dropout.default` | ✅ | ✅ | 드롭아웃 |
| `aten.native_dropout.default` | ✅ | ✅ | 네이티브 드롭아웃 |

**속성:**
- `p`: 드롭아웃 확률
- `training`: 학습 모드

## 14. 타입/메모리 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.to.dtype` | ✅ | ✅ | dtype 변환 |
| `aten.to.device` | ✅ | ✅ | device 변환 |
| `aten._to_copy.default` | ✅ | ✅ | 복사 변환 |
| `aten.contiguous.default` | ✅ | ✅ | 연속 메모리 |
| `aten.clone.default` | ✅ | ✅ | 복제 |

## 15. 인덱싱 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.select.int` | ✅ | ✅ | 인덱스 선택 |
| `aten.slice.Tensor` | ✅ | ✅ | 슬라이스 |
| `aten.index.Tensor` | ✅ | - | 텐서 인덱싱 |
| `aten.gather.default` | ✅ | ✅ | Gather |
| `aten.scatter.value` | ✅ | - | Scatter |

**Select 속성:**
- `dim`: 선택 차원
- `index`: 인덱스

**Slice 속성:**
- `dim`: 슬라이스 차원
- `start`, `end`, `step`: 범위

## 16. 비교 연산

| 연산자 | 변환 | 실행 | 설명 |
|--------|------|------|------|
| `aten.eq.Tensor` | ✅ | ✅ | 같음 |
| `aten.ne.Tensor` | ✅ | ✅ | 다름 |
| `aten.lt.Tensor` | ✅ | ✅ | 미만 |
| `aten.le.Tensor` | ✅ | - | 이하 |
| `aten.gt.Tensor` | ✅ | ✅ | 초과 |
| `aten.ge.Tensor` | ✅ | - | 이상 |
| `aten.where.self` | ✅ | ✅ | 조건 선택 |

## 17. 지원되지 않는 연산자 처리

지원되지 않는 연산자도 기본 변환기로 IR 추출이 가능합니다:

```python
# 기본 변환: 입출력 메타데이터는 보존
ir = extract_ir(model, inputs, strict=False)  # 오류 없이 추출

# Strict 모드: 미지원 연산자 시 오류
ir = extract_ir(model, inputs, strict=True)  # ConversionError 발생
```

## 18. 등록된 연산자 확인

```python
from npu_ir import list_registered_ops

ops = list_registered_ops()
print("Conversion ops:", len(ops['conversion']))
print("Execution ops:", len(ops['execution']))
```

## 19. 커스텀 연산자 추가

지원되지 않는 연산자는 직접 추가할 수 있습니다:

```python
from npu_ir.ops import register_op, register_executor
from npu_ir import OpNode

@register_op("aten.my_custom_op")
def convert_my_op(node_info):
    return OpNode(
        name=node_info.name,
        op_type="aten.my_custom_op",
        inputs=node_info.input_metas,
        outputs=node_info.output_metas,
        attrs=node_info.attrs,
    )

@register_executor("aten.my_custom_op")
def execute_my_op(inputs, attrs):
    # 실제 연산 수행
    result = my_custom_operation(inputs[0], **attrs)
    return [result]
```

자세한 내용은 [확장 가이드](extending.md)를 참조하세요.
