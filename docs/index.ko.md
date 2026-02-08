# NPU IR Extraction Framework

PyTorch 모델에서 컴파일러 백엔드용 IR(Intermediate Representation)을 추출하는 프레임워크입니다.

## 주요 특징

- **Weight-free 추출**: Meta tensor를 활용하여 실제 weight를 메모리에 로드하지 않고 그래프 구조만 추출
- **torch.export 기반**: PyTorch 공식 권장 방식인 TorchDynamo 기반 tracing 사용
- **완전한 메타데이터**: 모든 텐서의 shape, dtype 정보 자동 추출
- **IR 실행 및 검증**: 추출된 IR을 실행하여 원본 모델과 동일한 결과 검증 가능
- **확장 가능한 설계**: 커스텀 연산자 등록 메커니즘 제공
- **CLI 도구**: 터미널에서 IR 파일 조회 및 시각화 (`pytorch-ir info`, `pytorch-ir visualize`)

## 빠른 시작

### 설치

```bash
# uv 사용 (권장)
uv sync

# 또는 pip 사용
pip install -e .
```

### 기본 사용법

```python
import torch
from torch_ir import extract_ir, verify_ir_with_state_dict

# 1. Meta device에서 모델 생성
with torch.device('meta'):
    model = MyModel()
model.eval()

# 2. Example inputs 준비 (meta device)
example_inputs = (torch.randn(1, 3, 224, 224, device='meta'),)

# 3. IR 추출
ir = extract_ir(model, example_inputs)

# 4. IR 정보 확인
print(f"Nodes: {len(ir.nodes)}")
print(f"Weights: {len(ir.weights)}")
for node in ir.nodes[:5]:
    print(f"  {node.op_type}: {[t.shape for t in node.inputs]} -> {[t.shape for t in node.outputs]}")

# 5. IR 저장
ir.save("model_ir.json")
```

### IR 검증

```python
# 원본 모델과 IR 실행 결과 비교
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

## 문서

- [개념 및 아키텍처](concepts.md) - 프레임워크의 핵심 개념과 설계
- [환경 설정](setup.md) - 설치 및 개발 환경 구성
- [사용 가이드](usage.md) - 상세 사용법과 예제
- [API 레퍼런스](api/index.md) - 공개 API 문서
- [연산자 지원](operators.md) - 지원되는 ATen 연산자 목록
- [확장 가이드](extending.md) - 커스텀 연산자 추가 방법
- [CLI 레퍼런스](cli.ko.md) - IR 조회 및 시각화 커맨드라인 도구

## 의존성

- Python >= 3.10
- PyTorch >= 2.1

## 테스트 실행

```bash
# 기본 테스트
uv run pytest tests/ -v

# 종합 테스트 (모든 테스트 모델)
uv run pytest tests/test_comprehensive.py -v

# 리포트 생성
uv run pytest tests/test_comprehensive.py --generate-reports --output reports/
```
