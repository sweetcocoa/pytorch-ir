# 환경 설정

이 문서는 NPU IR 프레임워크의 설치 및 개발 환경 구성 방법을 설명합니다.

## 1. 요구 사항

### 1.1 시스템 요구 사항

- **Python**: 3.10 이상
- **운영체제**: Linux, macOS, Windows
- **메모리**: 최소 4GB RAM (대규모 모델 IR 추출 시 더 필요할 수 있음)

### 1.2 의존성

**필수 의존성:**
- `torch >= 2.1`

**개발 의존성:**
- `pytest >= 8.0`
- `torchvision >= 0.16` (테스트용)

**선택적 의존성:**
- `safetensors >= 0.4` (.safetensors 파일 지원)

## 2. 설치

### 2.1 uv 사용 (권장)

[uv](https://github.com/astral-sh/uv)는 빠른 Python 패키지 관리자입니다.

```bash
# uv 설치 (아직 없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 클론
git clone <repository-url>
cd my_compiler

# 의존성 설치 및 가상환경 생성
uv sync

# 개발 의존성 포함 설치
uv sync --dev
```

### 2.2 pip 사용

```bash
# 프로젝트 클론
git clone <repository-url>
cd my_compiler

# 가상환경 생성 (권장)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 또는
.venv\Scripts\activate     # Windows

# 패키지 설치
pip install -e .

# 개발 의존성 포함 설치
pip install -e ".[dev]"
```

### 2.3 safetensors 지원 설치

```bash
# uv 사용
uv add safetensors

# 또는 pip 사용
pip install safetensors
```

## 3. 설치 확인

### 3.1 기본 동작 확인

```python
import torch
from npu_ir import extract_ir

# 간단한 모델 테스트
with torch.device('meta'):
    model = torch.nn.Linear(10, 5)

inputs = (torch.randn(1, 10, device='meta'),)
ir = extract_ir(model, inputs, strict=False)

print(f"Nodes: {len(ir.nodes)}")
print(f"Weights: {len(ir.weights)}")
```

출력 예시:
```
Nodes: 1
Weights: 2
```

### 3.2 테스트 실행

```bash
# uv 사용
uv run pytest tests/ -v

# 또는 직접 실행
pytest tests/ -v
```

모든 테스트가 통과하면 설치가 올바르게 완료된 것입니다.

## 4. 개발 환경 구성

### 4.1 프로젝트 구조

```
my_compiler/
├── npu_ir/              # 메인 패키지
│   ├── __init__.py      # 공개 API
│   ├── ir.py            # IR 데이터 구조
│   ├── exporter.py      # torch.export 래핑
│   ├── analyzer.py      # 그래프 분석
│   ├── converter.py     # IR 변환
│   ├── serializer.py    # JSON 직렬화
│   ├── executor.py      # IR 실행기
│   ├── weight_loader.py # Weight 로더
│   ├── verifier.py      # 검증기
│   └── ops/             # 연산자 레지스트리 및 유틸리티
│       ├── __init__.py
│       ├── registry.py  # 커스텀 연산자 등록 메커니즘
│       ├── aten_ops.py  # op 타입 문자열 정규화 유틸리티
│       └── aten_impl.py # non-ATen op 실행 (getitem)
├── tests/               # 테스트 코드
│   ├── test_exporter.py
│   ├── test_analyzer.py
│   ├── test_converter.py
│   ├── test_executor.py
│   ├── test_verifier.py
│   └── test_models.py
├── examples/            # 예제 코드
│   ├── basic_usage.py
│   └── resnet_example.py
├── docs/                # 문서
├── pyproject.toml       # 프로젝트 설정
└── README.md
```

### 4.2 pyproject.toml

```toml
[project]
name = "npu-ir"
version = "0.1.0"
description = "PyTorch to NPU IR extraction framework"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "torchvision>=0.16",
]
safetensors = [
    "safetensors>=0.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
```

### 4.3 IDE 설정

#### VS Code

`.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.analysis.typeCheckingMode": "basic",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```

#### PyCharm

1. File → Settings → Project → Python Interpreter
2. `.venv` 가상환경 선택
3. Run → Edit Configurations → pytest 추가

## 5. 문제 해결

### 5.1 torch.export 오류

**증상**: `torch.export.export()` 호출 시 오류

**해결**:
```python
# PyTorch 버전 확인
import torch
print(torch.__version__)  # 2.1.0 이상이어야 함

# torch.export 사용 가능 확인
from torch.export import export
```

### 5.2 Meta device 오류

**증상**: `RuntimeError: meta tensors are not supported`

**해결**:
```python
# 올바른 meta tensor 생성
with torch.device('meta'):
    model = MyModel()

# 또는
model = MyModel()
model = model.to('meta')
```

### 5.3 Import 오류

**증상**: `ModuleNotFoundError: No module named 'npu_ir'`

**해결**:
```bash
# 개발 모드로 재설치
pip install -e .

# Python 경로 확인
python -c "import sys; print(sys.path)"
```

### 5.4 NumPy 경고

**증상**: `UserWarning: Failed to initialize NumPy`

**해결**:
```bash
# NumPy 설치 (선택사항, 경고만 있고 기능에 문제 없음)
pip install numpy
```

## 6. GPU 환경 (선택사항)

IR 추출은 meta device만 사용하므로 GPU가 필수는 아닙니다. 그러나 검증 시 GPU를 사용하면 속도가 향상됩니다.

### 6.1 CUDA 환경

```bash
# CUDA 버전에 맞는 PyTorch 설치
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 6.2 GPU 검증

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## 7. 다음 단계

환경 설정이 완료되면:

1. [사용 가이드](usage.md)를 읽고 기본 사용법 학습
2. [examples/](../examples/) 디렉토리의 예제 실행
3. [API 레퍼런스](api.md)에서 상세 API 확인
