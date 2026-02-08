"""IR Serializer for JSON serialization/deserialization."""

import json
from pathlib import Path
from typing import Any, Dict, Union

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

    # Check required fields
    if ir.nodes is None:
        errors.append("nodes is required")
    if ir.graph_inputs is None:
        errors.append("graph_inputs is required")
    if ir.graph_outputs is None:
        errors.append("graph_outputs is required")
    if ir.weights is None:
        errors.append("weights is required")

    # Validate nodes
    node_names = set()
    for i, node in enumerate(ir.nodes or []):
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

    for meta in ir.graph_inputs or []:
        validate_tensor_meta(meta, "graph_input")
    for meta in ir.graph_outputs or []:
        validate_tensor_meta(meta, "graph_output")
    for meta in ir.weights or []:
        validate_tensor_meta(meta, "weight")

    if errors:
        raise SerializationError(
            f"IR validation failed with {len(errors)} errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return True


def ir_to_dict(ir: IR) -> Dict[str, Any]:
    """Convert ``IR`` to a plain dictionary for custom serialization.

    Args:
        ir: The IR to convert.

    Returns:
        JSON-serializable dictionary.
    """
    return ir.to_dict()


def dict_to_ir(data: Dict[str, Any]) -> IR:
    """Convert a plain dictionary back to ``IR``.

    Args:
        data: Dictionary previously produced by ``ir_to_dict()``.

    Returns:
        Reconstructed ``IR`` instance.
    """
    return IR.from_dict(data)


class IRSerializer:
    """Class-based interface for IR serialization with optional validation."""

    def __init__(self, validate: bool = True):
        """Initialize the serializer.

        Args:
            validate: If True, validate IR before serialization.
        """
        self.validate = validate

    def serialize(self, ir: IR) -> str:
        """Serialize IR to JSON string.

        Args:
            ir: The IR to serialize.

        Returns:
            JSON string representation.
        """
        if self.validate:
            validate_ir(ir)
        return serialize_ir(ir)

    def deserialize(self, json_str: str) -> IR:
        """Deserialize IR from JSON string.

        Args:
            json_str: JSON string to parse.

        Returns:
            Deserialized ``IR`` instance.
        """
        ir = deserialize_ir(json_str)
        if self.validate:
            validate_ir(ir)
        return ir

    def save(self, ir: IR, path: Union[str, Path]) -> None:
        """Save IR to a JSON file.

        Args:
            ir: The IR to save.
            path: Output file path.
        """
        if self.validate:
            validate_ir(ir)
        save_ir(ir, path)

    def load(self, path: Union[str, Path]) -> IR:
        """Load IR from a JSON file.

        Args:
            path: Input file path.

        Returns:
            Deserialized ``IR`` instance.
        """
        ir = load_ir(path)
        if self.validate:
            validate_ir(ir)
        return ir
