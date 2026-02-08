"""Mermaid diagram generation from NPU IR — re-exported from npu_ir.visualize."""

from npu_ir.visualize import (
    _format_shape,
    _get_short_op_name,
    _sanitize_label,
    generate_op_distribution_pie,
    ir_to_mermaid,
)

__all__ = [
    "ir_to_mermaid",
    "generate_op_distribution_pie",
    "_format_shape",
    "_sanitize_label",
    "_get_short_op_name",
]
