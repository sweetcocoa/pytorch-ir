"""Operator registry for IR conversion and execution."""

from functools import wraps
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    pass


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


def _base_op_name(op_type: str) -> str:
    """Strip the overload suffix to get the base op name.

    For ATen ops the third dot-separated segment (the overload) is removed.
    Non-ATen strings are returned unchanged.

    Examples:
        >>> _base_op_name("aten.conv2d.default")
        'aten.conv2d'
        >>> _base_op_name("aten.add.Tensor")
        'aten.add'
        >>> _base_op_name("<built-in function getitem>")
        '<built-in function getitem>'

    Args:
        op_type: Fully-qualified op type string.

    Returns:
        Base op name without the overload suffix.
    """
    if op_type.startswith("aten."):
        parts = op_type.split(".")
        # aten.op_name.overload -> aten.op_name
        if len(parts) >= 3:
            return ".".join(parts[:2])
    return op_type


def _lookup_registry(registry: Dict[str, Callable], op_type: str) -> Optional[Callable]:
    """Look up a function in a registry with cascading match strategy.

    Tries in order: exact match, base-name match (strip overload suffix),
    then any registered pattern sharing the same base name.

    Args:
        registry: The registry dictionary to search.
        op_type: The op type string to look up.

    Returns:
        The registered function, or ``None`` if no match is found.
    """
    # Exact match
    if op_type in registry:
        return registry[op_type]

    # Match by base op name (strip overload suffix)
    base = _base_op_name(op_type)
    if base != op_type and base in registry:
        return registry[base]

    # Check if any registered pattern has the same base op name
    for pattern, fn in registry.items():
        if _base_op_name(pattern) == base:
            return fn

    return None


def get_conversion_fn(op_type: str) -> Optional[Callable]:
    """Get the conversion function for an operator type.

    Args:
        op_type: The operator type string

    Returns:
        The conversion function if registered, None otherwise.
    """
    return _lookup_registry(_CONVERSION_REGISTRY, op_type)


def get_execution_fn(op_type: str) -> Optional[Callable]:
    """Get the execution function for an operator type.

    Args:
        op_type: The operator type string

    Returns:
        The execution function if registered, None otherwise.
    """
    return _lookup_registry(_EXECUTION_REGISTRY, op_type)


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
