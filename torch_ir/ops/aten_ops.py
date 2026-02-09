"""ATen operator type normalization utilities."""

from typing import Any


def _normalize_op_type(target: Any) -> str:
    """Normalize the operator type to a standard ``aten.<op>.<overload>`` string format.

    Handles both ``torch.ops.aten.*`` and ``aten::*`` input formats.

    Args:
        target: The FX node target (function object or string).

    Returns:
        Normalized op type string (e.g., ``"aten.conv2d.default"``).
    """
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


def get_op_type(target: Any) -> str:
    """Get normalized operation type string from a target."""
    return _normalize_op_type(target)
