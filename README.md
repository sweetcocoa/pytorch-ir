# IR Extraction Framework

PyTorch 모델에서 컴파일러 백엔드용 IR(Intermediate Representation)을 추출하는 프레임워크입니다.

## 주요 특징

- **Weight-free 추출**: Meta tensor를 활용하여 실제 weight를 메모리에 로드하지 않고 그래프 구조만 추출
- **torch.export 기반**: PyTorch 공식 권장 방식인 TorchDynamo 기반 tracing 사용
- **완전한 메타데이터**: 모든 텐서의 shape, dtype 정보 자동 추출
- **IR 실행 및 검증**: 추출된 IR을 실행하여 원본 모델과 동일한 결과 검증 가능
- **확장 가능한 설계**: 커스텀 연산자 등록 메커니즘 제공

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

- [개념 및 아키텍처](docs/concepts.md) - 프레임워크의 핵심 개념과 설계
- [환경 설정](docs/setup.md) - 설치 및 개발 환경 구성
- [사용 가이드](docs/usage.md) - 상세 사용법과 예제
- [API 레퍼런스](docs/api/index.md) - 공개 API 문서
- [연산자 지원](docs/operators.md) - 지원되는 ATen 연산자 목록
- [확장 가이드](docs/extending.md) - 커스텀 연산자 추가 방법

## 프로젝트 구조

```
my_compiler/
├── torch_ir/
│   ├── __init__.py          # 공개 API
│   ├── ir.py                # IR 데이터 구조
│   ├── exporter.py          # torch.export 래핑
│   ├── analyzer.py          # 그래프 분석
│   ├── converter.py         # IR 변환
│   ├── serializer.py        # JSON 직렬화
│   ├── executor.py          # IR 실행기
│   ├── weight_loader.py     # Weight 로더
│   ├── verifier.py          # 검증기
│   └── ops/                 # 연산자 구현
├── tests/
│   ├── models/              # 테스트 모델 정의
│   │   ├── multi_io.py      # 다중 입출력 모델
│   │   ├── skip_connections.py  # 잔차 연결 모델
│   │   ├── shared_weights.py    # 가중치 공유 모델
│   │   └── attention.py     # 어텐션 모델
│   ├── testing/             # 테스트 프레임워크
│   │   ├── mermaid.py       # Mermaid DAG 생성
│   │   ├── statistics.py    # IR 통계 수집
│   │   ├── report.py        # 마크다운 리포트
│   │   └── runner.py        # 테스트 러너
│   ├── test_comprehensive.py  # 종합 테스트
│   └── __main__.py          # CLI 진입점
├── reports/                 # 테스트 리포트 (생성됨)
├── examples/                # 예제 코드
└── docs/                    # 문서
```

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

# 카테고리별 필터
uv run pytest tests/test_comprehensive.py -k "attention" -v

# CLI로 실행
uv run python -m tests --output reports/
uv run python -m tests --list-models
uv run python -m tests --category attention
```

## 라이선스

MIT License
