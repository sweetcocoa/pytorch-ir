"""ATen operator type normalization utilities."""

from typing import Any, Dict, Optional

from ..analyzer import NodeInfo
from ..ir import OpNode


def _normalize_op_type(target: Any) -> str:
    """Normalize the operator type to a standard string format."""
    target_str = str(target)

    # Handle torch.ops.aten.* format
    if "torch.ops.aten." in target_str:
        # Extract the operation name
        parts = target_str.replace("torch.ops.aten.", "aten.").split()
        return parts[0] if parts else target_str

    # Handle aten::* format
    if "aten::" in target_str:
        return target_str.replace("aten::", "aten.")

    return target_str


def _create_op_node(
    node_info: NodeInfo,
    op_type: Optional[str] = None,
    extra_attrs: Optional[Dict[str, Any]] = None,
) -> OpNode:
    """Helper to create an OpNode from NodeInfo."""
    final_op_type = op_type or _normalize_op_type(node_info.target)

    attrs = dict(node_info.attrs)
    if extra_attrs:
        attrs.update(extra_attrs)

    return OpNode(
        name=node_info.name,
        op_type=final_op_type,
        inputs=node_info.input_metas,
        outputs=node_info.output_metas,
        attrs=attrs,
    )


def get_op_type(target: Any) -> str:
    """Get normalized operation type string from a target."""
    return _normalize_op_type(target)
