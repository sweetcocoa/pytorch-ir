# 예제

간단한 MLP 모델의 IR 추출 결과를 PyTorch 소스 코드, JSON IR, 그래프 시각화로 비교합니다.

## SimpleMLP

`Linear(4, 8) → ReLU → Linear(8, 2)` 구조의 간단한 MLP 모델입니다.

=== "PyTorch 소스 코드"

    ```python
    import torch
    import torch.nn as nn
    from torch_ir import extract_ir

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(8, 2)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Meta device에서 모델 생성 및 IR 추출
    with torch.device('meta'):
        model = SimpleMLP()
    model.eval()

    example_inputs = (torch.randn(1, 4, device='meta'),)
    ir = extract_ir(model, example_inputs, model_name="SimpleMLP")

    print(f"노드 수: {len(ir.nodes)}")
    print(f"Weight 수: {len(ir.weights)}")
    for node in ir.nodes:
        print(f"  {node.op_type}: {[t.shape for t in node.outputs]}")
    ```

=== "JSON IR 출력"

    `ir.to_dict()` 호출 결과의 구조입니다:

    ```json
    {
      "model_name": "SimpleMLP",
      "pytorch_version": "2.x.x",
      "nodes": [
        {
          "name": "linear",
          "op_type": "aten.linear.default",
          "inputs": [
            {"name": "x", "shape": [1, 4], "dtype": "float32"},
            {"name": "p_fc1_weight", "shape": [8, 4], "dtype": "float32"},
            {"name": "p_fc1_bias", "shape": [8], "dtype": "float32"}
          ],
          "outputs": [
            {"name": "linear", "shape": [1, 8], "dtype": "float32"}
          ],
          "attrs": {}
        },
        {
          "name": "relu",
          "op_type": "aten.relu.default",
          "inputs": [
            {"name": "linear", "shape": [1, 8], "dtype": "float32"}
          ],
          "outputs": [
            {"name": "relu", "shape": [1, 8], "dtype": "float32"}
          ],
          "attrs": {}
        },
        {
          "name": "linear_1",
          "op_type": "aten.linear.default",
          "inputs": [
            {"name": "relu", "shape": [1, 8], "dtype": "float32"},
            {"name": "p_fc2_weight", "shape": [2, 8], "dtype": "float32"},
            {"name": "p_fc2_bias", "shape": [2], "dtype": "float32"}
          ],
          "outputs": [
            {"name": "linear_1", "shape": [1, 2], "dtype": "float32"}
          ],
          "attrs": {}
        }
      ],
      "graph_inputs": [
        {"name": "x", "shape": [1, 4], "dtype": "float32"}
      ],
      "graph_outputs": [
        {"name": "linear_1", "shape": [1, 2], "dtype": "float32"}
      ],
      "weights": [
        {"name": "p_fc1_weight", "shape": [8, 4], "dtype": "float32"},
        {"name": "p_fc1_bias", "shape": [8], "dtype": "float32"},
        {"name": "p_fc2_weight", "shape": [2, 8], "dtype": "float32"},
        {"name": "p_fc2_bias", "shape": [2], "dtype": "float32"}
      ],
      "weight_name_mapping": {
        "p_fc1_weight": "fc1.weight",
        "p_fc1_bias": "fc1.bias",
        "p_fc2_weight": "fc2.weight",
        "p_fc2_bias": "fc2.bias"
      }
    }
    ```

=== "IR 그래프 시각화"

    `ir_to_mermaid()` 함수로 생성한 Mermaid 다이어그램입니다. Weight 입력은 그래프 가독성을 위해 생략됩니다.

    ```mermaid
    flowchart TD
        input_x[/"Input: x<br/>1x4"/]
        op_linear["linear<br/>1x8"]
        op_relu["relu<br/>1x8"]
        op_linear_1["linear<br/>1x2"]
        output_0[\"Output<br/>1x2"/]
        input_x -->|"1x4"| op_linear
        op_linear -->|"1x8"| op_relu
        op_relu -->|"1x8"| op_linear_1
        op_linear_1 --> output_0
    ```

    **다이어그램 설명:**

    - 평행사변형 노드: 그래프 입력/출력
    - 사각형 노드: 연산 (op_type과 출력 shape 표시)
    - 엣지 라벨: 텐서 shape

## 프로그래밍적 시각화

`ir_to_mermaid()` 함수를 사용하면 임의의 IR을 Mermaid 다이어그램으로 변환할 수 있습니다:

```python
from torch_ir import extract_ir, ir_to_mermaid

# IR 추출
with torch.device('meta'):
    model = SimpleMLP()
model.eval()
ir = extract_ir(model, (torch.randn(1, 4, device='meta'),))

# Mermaid 다이어그램 생성
mermaid_str = ir_to_mermaid(ir)
print(mermaid_str)

# 대규모 모델의 경우 max_nodes로 노드 수 제한
mermaid_str = ir_to_mermaid(ir, max_nodes=20)
```

연산자 분포 파이 차트도 생성할 수 있습니다:

```python
from torch_ir import generate_op_distribution_pie

pie_chart = generate_op_distribution_pie(ir)
print(pie_chart)
```

자세한 API는 [Visualize API 레퍼런스](api/visualize.md)를 참조하세요.
