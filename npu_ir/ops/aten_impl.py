"""Non-ATen operator implementations for IR execution.

ATen ops are handled automatically by the schema-based fallback in executor.py.
Only non-ATen ops that cannot be resolved via torch.ops.aten need explicit executors.
"""

from typing import Any, Dict, List

import torch

from .registry import register_executor

TensorList = List[torch.Tensor]


@register_executor("<built-in function getitem>")
def execute_getitem(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute getitem — select one tensor from a multi-output op."""
    index = attrs.get("index", 0)
    return [inputs[index]]
