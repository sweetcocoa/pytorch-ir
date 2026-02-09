"""IR Serializer for JSON serialization/deserialization."""

import json
from pathlib import Path
from typing import Union

from .ir import IR, TensorMeta


class SerializationError(Exception):
    """Raised when serialization/deserialization fails."""

    pass


def serialize_ir(ir: IR) -> str:
    """Serialize IR to JSON string.

    Args:
        ir: The IR to serialize.

    Returns:
        JSON string representation.
    """
    return json.dumps(ir.to_dict(), indent=2)


def deserialize_ir(json_str: str) -> IR:
    """Deserialize IR from JSON string.

    Args:
        json_str: JSON string representation.

    Returns:
        The deserialized IR.

    Raises:
        SerializationError: If deserialization fails.
    """
    try:
        data = json.loads(json_str)
        return IR.from_dict(data)
    except json.JSONDecodeError as e:
        raise SerializationError(f"Invalid JSON: {e}") from e
    except (KeyError, TypeError) as e:
        raise SerializationError(f"Invalid IR format: {e}") from e


def save_ir(ir: IR, path: Union[str, Path]) -> None:
    """Save IR to a JSON file.

    Args:
        ir: The IR to save.
        path: The file path to save to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(serialize_ir(ir))


def load_ir(path: Union[str, Path]) -> IR:
    """Load IR from a JSON file.

    Args:
        path: The file path to load from.

    Returns:
        The loaded IR.

    Raises:
        SerializationError: If loading fails.
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"IR file not found: {path}")

    with open(path, "r") as f:
        return deserialize_ir(f.read())


def validate_ir(ir: IR) -> bool:
    """Validate the IR structure.

    Args:
        ir: The IR to validate.

    Returns:
        True if valid, raises exception otherwise.

    Raises:
        SerializationError: If validation fails.
    """
    errors = []

    # Validate nodes
    node_names = set()
    for i, node in enumerate(ir.nodes):
        if not node.name:
            errors.append(f"Node {i} missing name")
        elif node.name in node_names:
            errors.append(f"Duplicate node name: {node.name}")
        else:
            node_names.add(node.name)

        if not node.op_type:
            errors.append(f"Node '{node.name}' missing op_type")

    # Validate tensor metadata
    def validate_tensor_meta(meta: TensorMeta, context: str):
        if not meta.name:
            errors.append(f"{context}: missing name")
        if meta.shape is None:
            errors.append(f"{context} '{meta.name}': missing shape")
        if not meta.dtype:
            errors.append(f"{context} '{meta.name}': missing dtype")

    for meta in ir.graph_inputs:
        validate_tensor_meta(meta, "graph_input")
    for meta in ir.graph_outputs:
        validate_tensor_meta(meta, "graph_output")
    for meta in ir.weights:
        validate_tensor_meta(meta, "weight")

    if errors:
        raise SerializationError(
            f"IR validation failed with {len(errors)} errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return True
