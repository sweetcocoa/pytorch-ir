# 환경 설정

이 문서는 IR 추출 프레임워크의 개발 환경 구성 방법을 설명합니다.

!!! note
    기본 설치 방법은 [Home](index.md)을 참고하세요.

## 1. 개발 환경 구성

### 1.1 프로젝트 구조

```
my_compiler/
├── torch_ir/              # 메인 패키지
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

### 1.2 pyproject.toml

```toml
[project]
name = "torch-ir"
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

### 1.3 IDE 설정

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

## 2. 문제 해결

### 2.1 torch.export 오류

**증상**: `torch.export.export()` 호출 시 오류

**해결**:
```python
# PyTorch 버전 확인
import torch
print(torch.__version__)  # 2.1.0 이상이어야 함

# torch.export 사용 가능 확인
from torch.export import export
```

### 2.2 Meta device 오류

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

### 2.3 Import 오류

**증상**: `ModuleNotFoundError: No module named 'torch_ir'`

**해결**:
```bash
# 개발 모드로 재설치
pip install -e .

# Python 경로 확인
python -c "import sys; print(sys.path)"
```

### 2.4 NumPy 경고

**증상**: `UserWarning: Failed to initialize NumPy`

**해결**:
```bash
# NumPy 설치 (선택사항, 경고만 있고 기능에 문제 없음)
pip install numpy
```

## 3. GPU 환경 (선택사항)

IR 추출은 meta device만 사용하므로 GPU가 필수는 아닙니다. 그러나 검증 시 GPU를 사용하면 속도가 향상됩니다.

### 3.1 CUDA 환경

```bash
# CUDA 버전에 맞는 PyTorch 설치
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3.2 GPU 검증

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## 4. 다음 단계

환경 설정이 완료되면:

1. [사용 가이드](usage.md)를 읽고 기본 사용법 학습
2. [examples/](../examples/) 디렉토리의 예제 실행
3. [API 레퍼런스](api/index.md)에서 상세 API 확인
