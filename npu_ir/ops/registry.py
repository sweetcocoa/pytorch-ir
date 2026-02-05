"""Operator registry for IR conversion and execution."""

from typing import Callable, Dict, Any, Optional, TYPE_CHECKING
from functools import wraps

if TYPE_CHECKING:
    from ..ir import OpNode
    from ..analyzer import NodeInfo


# Registry for IR conversion functions (ATen op -> OpNode)
_CONVERSION_REGISTRY: Dict[str, Callable] = {}

# Registry for execution functions (op_type -> execution function)
_EXECUTION_REGISTRY: Dict[str, Callable] = {}


def register_op(op_pattern: str):
    """Decorator to register an IR conversion function for an operator.

    Args:
        op_pattern: The operator pattern to match (e.g., "aten.conv2d.default")

    Example:
        @register_op("aten.conv2d.default")
        def convert_conv2d(node_info: NodeInfo) -> OpNode:
            ...
    """

    def decorator(func: Callable) -> Callable:
        _CONVERSION_REGISTRY[op_pattern] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def register_executor(op_pattern: str):
    """Decorator to register an execution function for an operator.

    Args:
        op_pattern: The operator pattern to match (e.g., "aten.conv2d.default")

    Example:
        @register_executor("aten.conv2d.default")
        def execute_conv2d(inputs: List[torch.Tensor], attrs: Dict) -> List[torch.Tensor]:
            ...
    """

    def decorator(func: Callable) -> Callable:
        _EXECUTION_REGISTRY[op_pattern] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_conversion_fn(op_type: str) -> Optional[Callable]:
    """Get the conversion function for an operator type.

    Args:
        op_type: The operator type string

    Returns:
        The conversion function if registered, None otherwise.
    """
    # Try exact match first
    if op_type in _CONVERSION_REGISTRY:
        return _CONVERSION_REGISTRY[op_type]

    # Try pattern matching (e.g., "aten.conv2d" matches "aten.conv2d.default")
    for pattern, fn in _CONVERSION_REGISTRY.items():
        if pattern in op_type or op_type in pattern:
            return fn

    return None


def get_execution_fn(op_type: str) -> Optional[Callable]:
    """Get the execution function for an operator type.

    Args:
        op_type: The operator type string

    Returns:
        The execution function if registered, None otherwise.
    """
    # Try exact match first
    if op_type in _EXECUTION_REGISTRY:
        return _EXECUTION_REGISTRY[op_type]

    # Try pattern matching
    for pattern, fn in _EXECUTION_REGISTRY.items():
        if pattern in op_type or op_type in pattern:
            return fn

    return None


def list_registered_ops() -> Dict[str, list]:
    """List all registered operators.

    Returns:
        Dict with 'conversion' and 'execution' keys listing registered ops.
    """
    return {
        "conversion": list(_CONVERSION_REGISTRY.keys()),
        "execution": list(_EXECUTION_REGISTRY.keys()),
    }


def is_supported_op(op_type: str) -> bool:
    """Check if an operator is supported for conversion."""
    return get_conversion_fn(op_type) is not None


def clear_registry() -> None:
    """Clear all registered operators (useful for testing)."""
    _CONVERSION_REGISTRY.clear()
    _EXECUTION_REGISTRY.clear()
