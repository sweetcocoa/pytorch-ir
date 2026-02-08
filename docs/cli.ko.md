# CLI 레퍼런스

`torch-ir`는 Python 코드 작성 없이 터미널에서 IR JSON 파일을 조회하고 시각화할 수 있는 커맨드라인 인터페이스를 제공합니다.

## 설치

```bash
# 기본 설치
pip install torch-ir

# 이미지 렌더링 지원 (PNG/SVG)
pip install torch-ir[rendering]
```

## 커맨드

### `torch-ir --version`

설치된 버전을 출력합니다.

```bash
$ torch-ir --version
torch-ir 0.1.0
```

Python 모듈로도 실행할 수 있습니다:

```bash
python -m torch_ir --version
```

---

### `torch-ir info`

IR JSON 파일의 요약 정보를 표시합니다: 모델 이름, 노드/입력/출력/weight 수, 총 파라미터 수, shape, 연산자 분포 등.

```bash
torch-ir info model.json            # 텍스트 출력
torch-ir info model.json --json     # JSON 출력
torch-ir info model.json -o out.txt # 파일 저장
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `ir_file` (위치 인자) | IR JSON 파일 경로 | 필수 |
| `--json` | JSON 형식으로 출력 | `False` |
| `-o, --output FILE` | stdout 대신 파일로 출력 | `None` |

---

### `torch-ir visualize`

IR JSON 파일에서 Mermaid flowchart 다이어그램을 생성합니다.

```bash
torch-ir visualize model.json                  # stdout
torch-ir visualize model.json -o graph.mmd     # Mermaid 파일
torch-ir visualize model.json -o graph.png     # PNG 이미지
torch-ir visualize model.json -o graph.svg     # SVG 이미지
torch-ir visualize model.json --max-nodes 50   # 노드 수 제한
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `ir_file` (위치 인자) | IR JSON 파일 경로 | 필수 |
| `--max-nodes N` | 표시할 최대 노드 수 | `30` |
| `-o, --output FILE` | 출력 파일. `.png`/`.svg`는 이미지, 그 외는 Mermaid 텍스트 | `None` |

이미지 출력(`.png`, `.svg`)은 `rendering` optional dependency가 필요합니다:

```bash
pip install torch-ir[rendering]
```

---

## 예제

각 예제는 원본 PyTorch 모델 코드, `torch-ir info --json` 출력, `torch-ir visualize` 그래프를 탭으로 전환하며 확인할 수 있습니다.

---

### SimpleMLP

3-layer MLP. 가장 단순한 직선형 그래프 패턴입니다.

=== "PyTorch 모델"

    ```python
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    ```

=== "torch-ir info --json"

    ```json
    {
      "model_name": "SimpleMLP",
      "num_nodes": 5,
      "num_inputs": 1,
      "num_outputs": 1,
      "num_weights": 6,
      "total_parameters": 235146,
      "input_shapes": {
        "input": [1, 784]
      },
      "output_shapes": {
        "linear_2": [1, 10]
      },
      "op_distribution": {
        "aten.linear.default": 3,
        "aten.relu.default": 2
      }
    }
    ```

=== "torch-ir visualize"

    ```mermaid
    flowchart TD
        input_input[/"Input: input<br/>1x784"/]
        op_linear["linear<br/>1x256"]
        input_input -->|"1x784"| op_linear
        op_relu["relu<br/>1x256"]
        op_linear -->|"1x256"| op_relu
        op_linear_1["linear<br/>1x128"]
        op_relu -->|"1x256"| op_linear_1
        op_relu_1["relu<br/>1x128"]
        op_linear_1 -->|"1x128"| op_relu_1
        op_linear_2["linear<br/>1x10"]
        op_relu_1 -->|"1x128"| op_linear_2
        output_0[\"Output<br/>1x10"/]
        op_linear_2 --> output_0
    ```

---

### SelfAttention

Multi-head self-attention (4 heads, d_model=64). 입력에서 Q/K/V 프로젝션이 3개로 분기되고, `matmul → div → softmax → matmul` 패턴으로 어텐션을 수행합니다.

=== "PyTorch 모델"

    ```python
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SelfAttention(nn.Module):
        def __init__(self, d_model=64, num_heads=4):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.scale = math.sqrt(self.head_dim)
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            B, L, _ = x.shape
            q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.scale, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, -1)
            return self.out_proj(out)
    ```

=== "torch-ir info --json"

    ```json
    {
      "model_name": "SelfAttention",
      "num_nodes": 18,
      "num_inputs": 1,
      "num_outputs": 1,
      "num_weights": 8,
      "total_parameters": 16640,
      "input_shapes": {
        "x": [1, 16, 64]
      },
      "output_shapes": {
        "linear_3": [1, 16, 64]
      },
      "op_distribution": {
        "aten.transpose.int": 5,
        "aten.linear.default": 4,
        "aten.view.default": 4,
        "aten.matmul.default": 2,
        "aten.div.Tensor": 1,
        "aten.softmax.int": 1,
        "aten.contiguous.default": 1
      }
    }
    ```

=== "torch-ir visualize"

    ```mermaid
    flowchart TD
        input_x[/"Input: x<br/>1x16x64"/]
        op_linear["linear<br/>1x16x64"]
        input_x -->|"1x16x64"| op_linear
        op_linear_1["linear<br/>1x16x64"]
        input_x -->|"1x16x64"| op_linear_1
        op_linear_2["linear<br/>1x16x64"]
        input_x -->|"1x16x64"| op_linear_2
        op_view["view<br/>1x16x4x16"]
        op_linear -->|"1x16x64"| op_view
        op_transpose["transpose.int<br/>1x4x16x16"]
        op_view -->|"1x16x4x16"| op_transpose
        op_view_1["view<br/>1x16x4x16"]
        op_linear_1 -->|"1x16x64"| op_view_1
        op_transpose_1["transpose.int<br/>1x4x16x16"]
        op_view_1 -->|"1x16x4x16"| op_transpose_1
        op_view_2["view<br/>1x16x4x16"]
        op_linear_2 -->|"1x16x64"| op_view_2
        op_transpose_2["transpose.int<br/>1x4x16x16"]
        op_view_2 -->|"1x16x4x16"| op_transpose_2
        op_transpose_3["transpose.int<br/>1x4x16x16"]
        op_transpose_1 -->|"1x4x16x16"| op_transpose_3
        op_matmul["matmul<br/>1x4x16x16"]
        op_transpose -->|"1x4x16x16"| op_matmul
        op_transpose_3 -->|"1x4x16x16"| op_matmul
        op_div["div.Tensor<br/>1x4x16x16"]
        op_matmul -->|"1x4x16x16"| op_div
        op_softmax["softmax.int<br/>1x4x16x16"]
        op_div -->|"1x4x16x16"| op_softmax
        op_matmul_1["matmul<br/>1x4x16x16"]
        op_softmax -->|"1x4x16x16"| op_matmul_1
        op_transpose_2 -->|"1x4x16x16"| op_matmul_1
        op_transpose_4["transpose.int<br/>1x16x4x16"]
        op_matmul_1 -->|"1x4x16x16"| op_transpose_4
        op_contiguous["contiguous<br/>1x16x4x16"]
        op_transpose_4 -->|"1x16x4x16"| op_contiguous
        op_view_3["view<br/>1x16x64"]
        op_contiguous -->|"1x16x4x16"| op_view_3
        op_linear_3["linear<br/>1x16x64"]
        op_view_3 -->|"1x16x64"| op_linear_3
        output_0[\"Output<br/>1x16x64"/]
        op_linear_3 --> output_0
    ```

---

### MultiTaskHead

공유 CNN 백본에서 분류(10클래스)와 보조(4차원) 두 개의 출력 헤드가 분기되는 다중 출력 구조입니다. `flatten` 노드 이후 그래프가 두 갈래로 나뉘는 것을 확인할 수 있습니다.

=== "PyTorch 모델"

    ```python
    import torch.nn as nn

    class MultiTaskHead(nn.Module):
        def __init__(self, num_classes=10, aux_dim=4):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, num_classes),
            )
            self.aux_head = nn.Sequential(
                nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, aux_dim),
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features), self.aux_head(features)
    ```

=== "torch-ir info --json"

    ```json
    {
      "model_name": "MultiTaskHead",
      "num_nodes": 15,
      "num_inputs": 1,
      "num_outputs": 2,
      "num_weights": 22,
      "total_parameters": 218128,
      "input_shapes": {
        "x": [1, 3, 32, 32]
      },
      "output_shapes": {
        "linear_1": [1, 10],
        "linear_3": [1, 4]
      },
      "op_distribution": {
        "aten.relu.default": 4,
        "aten.linear.default": 4,
        "aten.conv2d.default": 2,
        "aten.batch_norm.default": 2,
        "aten.max_pool2d.default": 1,
        "aten.adaptive_avg_pool2d.default": 1,
        "aten.flatten.using_ints": 1
      }
    }
    ```

=== "torch-ir visualize"

    ```mermaid
    flowchart TD
        input_x[/"Input: x<br/>1x3x32x32"/]
        op_conv2d["conv2d<br/>1x32x32x32"]
        input_x -->|"1x3x32x32"| op_conv2d
        op_batch_norm["batch_norm<br/>1x32x32x32"]
        op_conv2d -->|"1x32x32x32"| op_batch_norm
        op_relu["relu<br/>1x32x32x32"]
        op_batch_norm -->|"1x32x32x32"| op_relu
        op_max_pool2d["max_pool2d<br/>1x32x16x16"]
        op_relu -->|"1x32x32x32"| op_max_pool2d
        op_conv2d_1["conv2d<br/>1x64x16x16"]
        op_max_pool2d -->|"1x32x16x16"| op_conv2d_1
        op_batch_norm_1["batch_norm<br/>1x64x16x16"]
        op_conv2d_1 -->|"1x64x16x16"| op_batch_norm_1
        op_relu_1["relu<br/>1x64x16x16"]
        op_batch_norm_1 -->|"1x64x16x16"| op_relu_1
        op_adaptive_avg_pool2d["adaptive_avg_pool2d<br/>1x64x4x4"]
        op_relu_1 -->|"1x64x16x16"| op_adaptive_avg_pool2d
        op_flatten["flatten.using_ints<br/>1x1024"]
        op_adaptive_avg_pool2d -->|"1x64x4x4"| op_flatten
        op_linear["linear<br/>1x128"]
        op_flatten -->|"1x1024"| op_linear
        op_relu_2["relu<br/>1x128"]
        op_linear -->|"1x128"| op_relu_2
        op_linear_1["linear<br/>1x10"]
        op_relu_2 -->|"1x128"| op_linear_1
        op_linear_2["linear<br/>1x64"]
        op_flatten -->|"1x1024"| op_linear_2
        op_relu_3["relu<br/>1x64"]
        op_linear_2 -->|"1x64"| op_relu_3
        op_linear_3["linear<br/>1x4"]
        op_relu_3 -->|"1x64"| op_linear_3
        output_0[\"Output<br/>1x10"/]
        op_linear_1 --> output_0
        output_1[\"Output<br/>1x4"/]
        op_linear_3 --> output_1
    ```

---

## Python 모듈로 실행

모든 커맨드는 `python -m torch_ir`로도 사용할 수 있습니다:

```bash
python -m torch_ir --version
python -m torch_ir info model.json
python -m torch_ir visualize model.json -o graph.png
```
