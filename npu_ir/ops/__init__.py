"""Operator registry and implementations.

This module provides:
- registry: Operator registration for IR conversion and execution
- aten_ops: ATen operator mappings for IR conversion
- aten_impl: ATen operator implementations for IR execution

Custom operators can be registered using decorators:

    from npu_ir.ops import register_op, register_executor

    @register_op("custom.my_op")
    def convert_my_op(node_info):
        return OpNode(...)

    @register_executor("custom.my_op")
    def execute_my_op(inputs, attrs):
        return [result_tensor]
"""

from .registry import (
    register_op,
    register_executor,
    get_conversion_fn,
    get_execution_fn,
    list_registered_ops,
    is_supported_op,
)

__all__ = [
    "register_op",
    "register_executor",
    "get_conversion_fn",
    "get_execution_fn",
    "list_registered_ops",
    "is_supported_op",
]
