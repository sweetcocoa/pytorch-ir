# API 레퍼런스

이 문서는 NPU IR 프레임워크의 공개 API를 설명합니다.

## 1. 주요 함수

### extract_ir

```python
def extract_ir(
    model: nn.Module,
    example_inputs: Tuple[Any, ...],
    *,
    model_name: Optional[str] = None,
    strict: bool = True,
) -> NPU_IR
```

PyTorch 모델에서 NPU IR을 추출합니다.

**파라미터:**
- `model`: 추출할 PyTorch 모델 (meta device에 있어야 함)
- `example_inputs`: 트레이싱용 예제 입력 (meta device에 있어야 함)
- `model_name`: 모델 이름 (None이면 클래스 이름 사용)
- `strict`: True면 meta device 검증, 미지원 연산자 시 오류

**반환값:**
- `NPU_IR`: 추출된 IR

**예외:**
- `ExportError`: 모델 export 실패 시
- `ConversionError`: IR 변환 실패 시 (strict 모드)

**예시:**
```python
with torch.device('meta'):
    model = nn.Linear(10, 5)
model.eval()
inputs = (torch.randn(1, 10, device='meta'),)
ir = extract_ir(model, inputs)
```

---

### verify_ir

```python
def verify_ir(
    ir: NPU_IR,
    weights_path: Union[str, Path],
    original_model: nn.Module,
    test_inputs: Tuple[torch.Tensor, ...],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Tuple[bool, VerificationReport]
```

IR 실행 결과와 원본 모델 출력을 비교합니다.

**파라미터:**
- `ir`: 검증할 IR
- `weights_path`: weight 파일 경로 (.pt 또는 .safetensors)
- `original_model`: 비교할 원본 모델 (weight 로드됨)
- `test_inputs`: 테스트 입력 텐서
- `rtol`: 상대 오차 허용치
- `atol`: 절대 오차 허용치

**반환값:**
- `Tuple[bool, VerificationReport]`: (검증 성공 여부, 상세 리포트)

---

### verify_ir_with_state_dict

```python
def verify_ir_with_state_dict(
    ir: NPU_IR,
    state_dict: Dict[str, torch.Tensor],
    original_model: nn.Module,
    test_inputs: Tuple[torch.Tensor, ...],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Tuple[bool, VerificationReport]
```

state_dict를 사용하여 IR을 검증합니다.

**파라미터:**
- `state_dict`: weight state dictionary

나머지 파라미터는 `verify_ir`과 동일

---

### execute_ir

```python
def execute_ir(
    ir: NPU_IR,
    inputs: Tuple[torch.Tensor, ...],
    weights: Optional[Dict[str, torch.Tensor]] = None,
    weights_path: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, ...]
```

IR 그래프를 실행합니다.

**파라미터:**
- `ir`: 실행할 IR
- `inputs`: 입력 텐서
- `weights`: weight dictionary (weights_path와 배타적)
- `weights_path`: weight 파일 경로

**반환값:**
- `Tuple[torch.Tensor, ...]`: 출력 텐서들

**예외:**
- `ExecutionError`: 실행 실패 시
- `ValueError`: weights와 weights_path 둘 다 없을 때

---

## 2. 데이터 클래스

### TensorMeta

```python
@dataclass
class TensorMeta:
    name: str                 # 텐서 이름
    shape: Tuple[int, ...]    # Shape
    dtype: str                # "float32", "float16" 등
```

텐서의 메타데이터 (실제 데이터 없음)

**메서드:**
- `to_dict() -> Dict[str, Any]`: 딕셔너리로 변환
- `from_dict(data: Dict) -> TensorMeta`: 딕셔너리에서 생성 (classmethod)

---

### OpNode

```python
@dataclass
class OpNode:
    name: str                       # 노드 고유 이름
    op_type: str                    # "aten.conv2d.default" 등
    inputs: List[TensorMeta]        # 입력 텐서 메타데이터
    outputs: List[TensorMeta]       # 출력 텐서 메타데이터
    attrs: Dict[str, Any] = {}      # 연산 속성
```

IR 그래프의 연산 노드

**메서드:**
- `to_dict() -> Dict[str, Any]`: 딕셔너리로 변환
- `from_dict(data: Dict) -> OpNode`: 딕셔너리에서 생성 (classmethod)

---

### NPU_IR

```python
@dataclass
class NPU_IR:
    nodes: List[OpNode]                      # 연산 노드 리스트
    graph_inputs: List[TensorMeta]           # 그래프 입력
    graph_outputs: List[TensorMeta]          # 그래프 출력
    weights: List[TensorMeta]                # Weight 메타데이터
    weight_name_mapping: Dict[str, str] = {} # placeholder → state_dict 키
    model_name: str = ""                     # 모델 이름
    pytorch_version: str = ""                # PyTorch 버전
```

완전한 IR 표현

**메서드:**
- `to_dict() -> Dict[str, Any]`: 딕셔너리로 변환
- `from_dict(data: Dict) -> NPU_IR`: 딕셔너리에서 생성 (classmethod)
- `save(path: str) -> None`: JSON 파일로 저장
- `load(path: str) -> NPU_IR`: JSON 파일에서 로드 (classmethod)

---

### VerificationReport

```python
@dataclass
class VerificationReport:
    is_valid: bool                           # 검증 성공 여부
    max_diff: float = 0.0                    # 최대 차이
    mean_diff: float = 0.0                   # 평균 차이
    num_outputs: int = 0                     # 출력 개수
    output_details: List[Dict[str, Any]] = [] # 개별 출력 상세
    error_message: Optional[str] = None      # 오류 메시지
```

검증 결과 리포트

---

## 3. 직렬화 함수

### serialize_ir / deserialize_ir

```python
def serialize_ir(ir: NPU_IR) -> str
def deserialize_ir(json_str: str) -> NPU_IR
```

IR을 JSON 문자열로 직렬화/역직렬화

---

### save_ir / load_ir

```python
def save_ir(ir: NPU_IR, path: Union[str, Path]) -> None
def load_ir(path: Union[str, Path]) -> NPU_IR
```

IR을 JSON 파일로 저장/로드

---

### validate_ir

```python
def validate_ir(ir: NPU_IR) -> bool
```

IR 구조 검증. 유효하면 True, 아니면 SerializationError 발생

---

## 4. Weight 로더

### load_weights

```python
def load_weights(path: Union[str, Path]) -> Dict[str, torch.Tensor]
```

자동 포맷 감지하여 weight 로드

---

### load_weights_pt / load_weights_safetensors

```python
def load_weights_pt(path: Union[str, Path]) -> Dict[str, torch.Tensor]
def load_weights_safetensors(path: Union[str, Path]) -> Dict[str, torch.Tensor]
```

특정 포맷으로 weight 로드

---

## 5. 클래스 기반 API

### IRExecutor

```python
class IRExecutor:
    def __init__(self, ir: NPU_IR, weights: Optional[Dict[str, torch.Tensor]] = None)
    def load_weights(self, path: Union[str, Path]) -> None
    def load_weights_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None
    def execute(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]
    def __call__(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]
```

IR 실행기

**예시:**
```python
executor = IRExecutor(ir)
executor.load_weights_from_state_dict(state_dict)
outputs = executor.execute((test_input,))
# 또는
outputs = executor(test_input)
```

---

### IRConverter

```python
class IRConverter:
    def __init__(self, strict: bool = False)
    def convert(self, exported: ExportedProgram, model_name: str = "") -> NPU_IR
```

ExportedProgram을 NPU_IR로 변환

---

### IRSerializer

```python
class IRSerializer:
    def __init__(self, validate: bool = True)
    def serialize(self, ir: NPU_IR) -> str
    def deserialize(self, json_str: str) -> NPU_IR
    def save(self, ir: NPU_IR, path: Union[str, Path]) -> None
    def load(self, path: Union[str, Path]) -> NPU_IR
```

검증 옵션이 있는 직렬화기

---

### IRVerifier

```python
class IRVerifier:
    def __init__(self, rtol: float = 1e-5, atol: float = 1e-5)
    def verify(self, ir, weights_path, original_model, test_inputs) -> Tuple[bool, VerificationReport]
    def verify_with_state_dict(self, ir, state_dict, original_model, test_inputs) -> Tuple[bool, VerificationReport]
```

클래스 기반 검증기

---

### WeightLoader

```python
class WeightLoader:
    def __init__(self, validate: bool = True)
    def load(self, path: Union[str, Path], ir: Optional[NPU_IR] = None) -> Dict[str, torch.Tensor]
    def load_from_state_dict(self, state_dict, ir: Optional[NPU_IR] = None) -> Dict[str, torch.Tensor]
```

검증 옵션이 있는 weight 로더

---

## 6. 연산자 등록

### register_op

```python
@register_op("aten.custom_op")
def convert_custom_op(node_info: NodeInfo) -> OpNode:
    ...
```

IR 변환 함수 등록 데코레이터

---

### register_executor

```python
@register_executor("aten.custom_op")
def execute_custom_op(inputs: List[torch.Tensor], attrs: Dict[str, Any]) -> List[torch.Tensor]:
    ...
```

실행 함수 등록 데코레이터

---

### list_registered_ops

```python
def list_registered_ops() -> Dict[str, list]
```

등록된 연산자 목록 반환

**반환값:**
```python
{
    "conversion": ["aten.conv2d.default", "aten.linear.default", ...],
    "execution": ["aten.conv2d.default", "aten.linear.default", ...]
}
```

---

## 7. 예외 클래스

### ExportError

모델 export 실패 시 발생

```python
from npu_ir import ExportError

try:
    ir = extract_ir(model, inputs)
except ExportError as e:
    print(f"Export failed: {e}")
```

---

### ConversionError

IR 변환 실패 시 발생 (strict 모드)

```python
from npu_ir import ConversionError
```

---

### ExecutionError

IR 실행 실패 시 발생

```python
from npu_ir import ExecutionError
```

---

### SerializationError

직렬화/역직렬화 실패 시 발생

```python
from npu_ir import SerializationError
```

---

### WeightLoadError

weight 로드 실패 시 발생

```python
from npu_ir import WeightLoadError
```

---

## 8. 저수준 API

### export_model

```python
def export_model(
    model: nn.Module,
    example_inputs: Tuple[Any, ...],
    *,
    strict: bool = True,
) -> ExportedProgram
```

torch.export.export() 래퍼

---

### GraphAnalyzer

```python
class GraphAnalyzer:
    def __init__(self, exported: ExportedProgram)
    def get_graph_inputs(self) -> List[TensorMeta]
    def get_graph_outputs(self) -> List[TensorMeta]
    def get_weights(self) -> List[TensorMeta]
    def get_weight_name_mapping(self) -> Dict[str, str]
    def get_call_function_nodes(self) -> List[NodeInfo]
    def get_node_output_meta(self, node_name: str) -> List[TensorMeta]
    def get_node_input_meta(self, node: Node) -> List[TensorMeta]
```

ExportedProgram 분석기

---

### NodeInfo

```python
@dataclass
class NodeInfo:
    name: str                       # 노드 이름
    op: str                         # "call_function" 등
    target: Any                     # 연산 대상
    args: Tuple[Any, ...]           # 인자
    kwargs: Dict[str, Any]          # 키워드 인자
    input_metas: List[TensorMeta]   # 입력 메타데이터
    output_metas: List[TensorMeta]  # 출력 메타데이터
    attrs: Dict[str, Any]           # 추출된 속성
```

FX 노드에서 추출한 정보

---

## 9. 전체 Import 목록

```python
from npu_ir import (
    # 주요 함수
    extract_ir,
    verify_ir,
    verify_ir_with_state_dict,
    execute_ir,

    # 데이터 클래스
    NPU_IR,
    OpNode,
    TensorMeta,
    VerificationReport,

    # 직렬화
    serialize_ir,
    deserialize_ir,
    save_ir,
    load_ir,
    validate_ir,

    # Weight
    load_weights,
    load_weights_pt,
    load_weights_safetensors,

    # 클래스
    IRExecutor,
    IRConverter,
    IRSerializer,
    IRVerifier,
    WeightLoader,

    # 연산자 등록
    register_op,
    register_executor,
    list_registered_ops,

    # 예외
    ExportError,
    ConversionError,
    ExecutionError,
    SerializationError,
    WeightLoadError,

    # 저수준
    export_model,
)
```
