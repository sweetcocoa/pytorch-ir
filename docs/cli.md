# CLI Reference

`pytorch-ir` provides a command-line interface for inspecting and visualizing IR JSON files without writing Python code.

## Installation

```bash
# Basic installation
pip install pytorch-ir

# With image rendering support (PNG/SVG)
pip install pytorch-ir[rendering]
```

## Commands

### `pytorch-ir --version`

Print the installed version.

```bash
$ pytorch-ir --version
pytorch-ir 0.1.0
```

You can also run the CLI as a Python module:

```bash
python -m torch_ir --version
```

---

### `pytorch-ir info`

Display a summary of an IR JSON file: model name, node/input/output/weight counts, total parameters, shapes, and operator distribution.

```bash
pytorch-ir info model.json            # text output
pytorch-ir info model.json --json     # JSON output
pytorch-ir info model.json -o out.txt # save to file
```

| Option | Description | Default |
|--------|-------------|---------|
| `ir_file` (positional) | Path to IR JSON file | Required |
| `--json` | Output in JSON format | `False` |
| `-o, --output FILE` | Write output to file instead of stdout | `None` |

---

### `pytorch-ir visualize`

Generate a Mermaid flowchart diagram from an IR JSON file.

```bash
pytorch-ir visualize model.json                  # stdout
pytorch-ir visualize model.json -o graph.mmd     # Mermaid file
pytorch-ir visualize model.json -o graph.png     # PNG image
pytorch-ir visualize model.json -o graph.svg     # SVG image
pytorch-ir visualize model.json --max-nodes 50   # limit nodes
```

| Option | Description | Default |
|--------|-------------|---------|
| `ir_file` (positional) | Path to IR JSON file | Required |
| `--max-nodes N` | Maximum number of nodes to display | `30` |
| `--no-weights` | Hide weight inputs from the diagram | `False` |
| `-o, --output FILE` | Output file. `.png`/`.svg` for image, others for Mermaid text | `None` |

Image output (`.png`, `.svg`) requires the `rendering` optional dependency:

```bash
pip install pytorch-ir[rendering]
```

---

## Examples

Each example shows the original PyTorch model code, `pytorch-ir info --json` output, and `pytorch-ir visualize` graph.

---

### SimpleMLP

A 3-layer MLP. The simplest linear graph pattern.

=== "PyTorch Model"

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

=== "pytorch-ir info --json"

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

=== "pytorch-ir visualize"

    ```mermaid
    flowchart TD
        input_input[/"Input: input<br/>1x784"/]
        op_linear["linear<br/>1x256"]
        input_input -->|"1x784"| op_linear
        w_p_0_weight[/"p_0_weight<br/>256x784"/]
        w_p_0_weight -.->|"256x784"| op_linear
        w_p_0_bias[/"p_0_bias<br/>256"/]
        w_p_0_bias -.->|"256"| op_linear
        op_relu["relu<br/>1x256"]
        op_linear -->|"1x256"| op_relu
        op_linear_1["linear<br/>1x128"]
        op_relu -->|"1x256"| op_linear_1
        w_p_2_weight[/"p_2_weight<br/>128x256"/]
        w_p_2_weight -.->|"128x256"| op_linear_1
        w_p_2_bias[/"p_2_bias<br/>128"/]
        w_p_2_bias -.->|"128"| op_linear_1
        op_relu_1["relu<br/>1x128"]
        op_linear_1 -->|"1x128"| op_relu_1
        op_linear_2["linear<br/>1x10"]
        op_relu_1 -->|"1x128"| op_linear_2
        w_p_4_weight[/"p_4_weight<br/>10x128"/]
        w_p_4_weight -.->|"10x128"| op_linear_2
        w_p_4_bias[/"p_4_bias<br/>10"/]
        w_p_4_bias -.->|"10"| op_linear_2
        output_0[\"Output<br/>1x10"/]
        op_linear_2 --> output_0
    ```

---

### SelfAttention

Multi-head self-attention (4 heads, d_model=64). The input fans out into 3 parallel Q/K/V projections, followed by the `matmul → div → softmax → matmul` attention pattern.

=== "PyTorch Model"

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

=== "pytorch-ir info --json"

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

=== "pytorch-ir visualize"

    ```mermaid
    flowchart TD
        input_x[/"Input: x<br/>1x16x64"/]
        op_linear["linear<br/>1x16x64"]
        input_x -->|"1x16x64"| op_linear
        w_p_q_proj_weight[/"p_q_proj_weight<br/>64x64"/]
        w_p_q_proj_weight -.->|"64x64"| op_linear
        w_p_q_proj_bias[/"p_q_proj_bias<br/>64"/]
        w_p_q_proj_bias -.->|"64"| op_linear
        op_view["view<br/>1x16x4x16"]
        op_linear -->|"1x16x64"| op_view
        op_transpose["transpose.int<br/>1x4x16x16"]
        op_view -->|"1x16x4x16"| op_transpose
        op_linear_1["linear<br/>1x16x64"]
        input_x -->|"1x16x64"| op_linear_1
        w_p_k_proj_weight[/"p_k_proj_weight<br/>64x64"/]
        w_p_k_proj_weight -.->|"64x64"| op_linear_1
        w_p_k_proj_bias[/"p_k_proj_bias<br/>64"/]
        w_p_k_proj_bias -.->|"64"| op_linear_1
        op_view_1["view<br/>1x16x4x16"]
        op_linear_1 -->|"1x16x64"| op_view_1
        op_transpose_1["transpose.int<br/>1x4x16x16"]
        op_view_1 -->|"1x16x4x16"| op_transpose_1
        op_linear_2["linear<br/>1x16x64"]
        input_x -->|"1x16x64"| op_linear_2
        w_p_v_proj_weight[/"p_v_proj_weight<br/>64x64"/]
        w_p_v_proj_weight -.->|"64x64"| op_linear_2
        w_p_v_proj_bias[/"p_v_proj_bias<br/>64"/]
        w_p_v_proj_bias -.->|"64"| op_linear_2
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
        w_p_out_proj_weight[/"p_out_proj_weight<br/>64x64"/]
        w_p_out_proj_weight -.->|"64x64"| op_linear_3
        w_p_out_proj_bias[/"p_out_proj_bias<br/>64"/]
        w_p_out_proj_bias -.->|"64"| op_linear_3
        output_0[\"Output<br/>1x16x64"/]
        op_linear_3 --> output_0
    ```

---

### MultiTaskHead

A shared CNN backbone that branches into two output heads — classification (10 classes) and auxiliary (4-dim). Notice how the graph forks after the `flatten` node.

=== "PyTorch Model"

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

=== "pytorch-ir info --json"

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

=== "pytorch-ir visualize"

    ```mermaid
    flowchart TD
        input_x[/"Input: x<br/>1x3x32x32"/]
        op_conv2d["conv2d<br/>1x32x32x32"]
        input_x -->|"1x3x32x32"| op_conv2d
        w_p_backbone_0_weight[/"p_backbone_0_weight<br/>32x3x3x3"/]
        w_p_backbone_0_weight -.->|"32x3x3x3"| op_conv2d
        w_p_backbone_0_bias[/"p_backbone_0_bias<br/>32"/]
        w_p_backbone_0_bias -.->|"32"| op_conv2d
        op_batch_norm["batch_norm<br/>1x32x32x32"]
        op_conv2d -->|"1x32x32x32"| op_batch_norm
        w_p_backbone_1_weight[/"p_backbone_1_weight<br/>32"/]
        w_p_backbone_1_weight -.->|"32"| op_batch_norm
        w_p_backbone_1_bias[/"p_backbone_1_bias<br/>32"/]
        w_p_backbone_1_bias -.->|"32"| op_batch_norm
        w_b_backbone_1_running_mean[/"b_backbone_1_running_mean<br/>32"/]
        w_b_backbone_1_running_mean -.->|"32"| op_batch_norm
        w_b_backbone_1_running_var[/"b_backbone_1_running_var<br/>32"/]
        w_b_backbone_1_running_var -.->|"32"| op_batch_norm
        op_relu["relu<br/>1x32x32x32"]
        op_batch_norm -->|"1x32x32x32"| op_relu
        op_max_pool2d["max_pool2d<br/>1x32x16x16"]
        op_relu -->|"1x32x32x32"| op_max_pool2d
        op_conv2d_1["conv2d<br/>1x64x16x16"]
        op_max_pool2d -->|"1x32x16x16"| op_conv2d_1
        w_p_backbone_4_weight[/"p_backbone_4_weight<br/>64x32x3x3"/]
        w_p_backbone_4_weight -.->|"64x32x3x3"| op_conv2d_1
        w_p_backbone_4_bias[/"p_backbone_4_bias<br/>64"/]
        w_p_backbone_4_bias -.->|"64"| op_conv2d_1
        op_batch_norm_1["batch_norm<br/>1x64x16x16"]
        op_conv2d_1 -->|"1x64x16x16"| op_batch_norm_1
        w_p_backbone_5_weight[/"p_backbone_5_weight<br/>64"/]
        w_p_backbone_5_weight -.->|"64"| op_batch_norm_1
        w_p_backbone_5_bias[/"p_backbone_5_bias<br/>64"/]
        w_p_backbone_5_bias -.->|"64"| op_batch_norm_1
        w_b_backbone_5_running_mean[/"b_backbone_5_running_mean<br/>64"/]
        w_b_backbone_5_running_mean -.->|"64"| op_batch_norm_1
        w_b_backbone_5_running_var[/"b_backbone_5_running_var<br/>64"/]
        w_b_backbone_5_running_var -.->|"64"| op_batch_norm_1
        op_relu_1["relu<br/>1x64x16x16"]
        op_batch_norm_1 -->|"1x64x16x16"| op_relu_1
        op_adaptive_avg_pool2d["adaptive_avg_pool2d<br/>1x64x4x4"]
        op_relu_1 -->|"1x64x16x16"| op_adaptive_avg_pool2d
        op_flatten["flatten.using_ints<br/>1x1024"]
        op_adaptive_avg_pool2d -->|"1x64x4x4"| op_flatten
        op_linear["linear<br/>1x128"]
        op_flatten -->|"1x1024"| op_linear
        w_p_classifier_0_weight[/"p_classifier_0_weight<br/>128x1024"/]
        w_p_classifier_0_weight -.->|"128x1024"| op_linear
        w_p_classifier_0_bias[/"p_classifier_0_bias<br/>128"/]
        w_p_classifier_0_bias -.->|"128"| op_linear
        op_relu_2["relu<br/>1x128"]
        op_linear -->|"1x128"| op_relu_2
        op_linear_1["linear<br/>1x10"]
        op_relu_2 -->|"1x128"| op_linear_1
        w_p_classifier_2_weight[/"p_classifier_2_weight<br/>10x128"/]
        w_p_classifier_2_weight -.->|"10x128"| op_linear_1
        w_p_classifier_2_bias[/"p_classifier_2_bias<br/>10"/]
        w_p_classifier_2_bias -.->|"10"| op_linear_1
        op_linear_2["linear<br/>1x64"]
        op_flatten -->|"1x1024"| op_linear_2
        w_p_aux_head_0_weight[/"p_aux_head_0_weight<br/>64x1024"/]
        w_p_aux_head_0_weight -.->|"64x1024"| op_linear_2
        w_p_aux_head_0_bias[/"p_aux_head_0_bias<br/>64"/]
        w_p_aux_head_0_bias -.->|"64"| op_linear_2
        op_relu_3["relu<br/>1x64"]
        op_linear_2 -->|"1x64"| op_relu_3
        op_linear_3["linear<br/>1x4"]
        op_relu_3 -->|"1x64"| op_linear_3
        w_p_aux_head_2_weight[/"p_aux_head_2_weight<br/>4x64"/]
        w_p_aux_head_2_weight -.->|"4x64"| op_linear_3
        w_p_aux_head_2_bias[/"p_aux_head_2_bias<br/>4"/]
        w_p_aux_head_2_bias -.->|"4"| op_linear_3
        output_0[\"Output<br/>1x10"/]
        op_linear_1 --> output_0
        output_1[\"Output<br/>1x4"/]
        op_linear_3 --> output_1
    ```

---

## Usage with Python module

All commands are also available via `python -m torch_ir`:

```bash
python -m torch_ir --version
python -m torch_ir info model.json
python -m torch_ir visualize model.json -o graph.png
```
