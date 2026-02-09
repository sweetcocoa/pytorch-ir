"""IR extraction framework for PyTorch models.

This framework extracts IR from PyTorch models using torch.export,
without loading actual weight values (using meta tensors).

Example usage:
    import torch
    from torch_ir import extract_ir, verify_ir

    # Create model on meta device
    with torch.device('meta'):
        model = MyModel()

    # Extract IR
    example_inputs = (torch.randn(1, 3, 224, 224, device='meta'),)
    ir = extract_ir(model, example_inputs)

    # Save IR
    ir.save("model.json")

    # Verify IR (with actual weights)
    is_valid, report = verify_ir(
        ir=ir,
        weights_path="model_weights.pt",
        original_model=original_model,
        test_inputs=(torch.randn(1, 3, 224, 224),),
    )
"""

from typing import Any, Optional, Tuple

import torch.nn as nn

from .converter import ConversionError, convert_exported_program
from .executor import ExecutionError, IRExecutor, execute_ir
from .exporter import ExportError, export_model, get_model_name
from .ir import IR, OpNode, TensorMeta

# Import aten ops and implementations to register them
from .ops import aten_impl, aten_ops  # noqa: F401
from .ops.registry import register_executor, register_op
from .serializer import (
    SerializationError,
    deserialize_ir,
    load_ir,
    save_ir,
    serialize_ir,
    validate_ir,
)
from .verifier import VerificationReport, verify_ir, verify_ir_with_state_dict
from .visualize import generate_op_distribution_pie, ir_to_mermaid
from .weight_loader import (
    WeightLoadError,
    load_weights,
    load_weights_pt,
    load_weights_safetensors,
)

try:
    from torch_ir._version import __version__
except ModuleNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    # Main API
    "extract_ir",
    "verify_ir",
    "verify_ir_with_state_dict",
    # IR data structures
    "IR",
    "OpNode",
    "TensorMeta",
    # Exporter
    "export_model",
    "ExportError",
    # Converter
    "convert_exported_program",
    "ConversionError",
    # Serialization
    "serialize_ir",
    "deserialize_ir",
    "save_ir",
    "load_ir",
    "validate_ir",
    "SerializationError",
    # Execution
    "IRExecutor",
    "execute_ir",
    "ExecutionError",
    # Weights
    "load_weights",
    "load_weights_pt",
    "load_weights_safetensors",
    "WeightLoadError",
    # Verification
    "VerificationReport",
    # Op registration
    "register_op",
    "register_executor",
    # Visualization
    "ir_to_mermaid",
    "generate_op_distribution_pie",
]


def extract_ir(
    model: nn.Module,
    example_inputs: Tuple[Any, ...],
    *,
    model_name: Optional[str] = None,
    strict: bool = True,
) -> IR:
    """Extract IR from a PyTorch model.

    This is the main entry point for IR extraction. The model should be on
    meta device for weight-free extraction.

    Args:
        model: The PyTorch model to extract IR from. Should be on meta device.
        example_inputs: Example inputs for tracing. Should be on meta device.
        model_name: Optional name for the model. If None, uses class name.
        strict: If True, validate meta device and raise errors for unsupported ops.

    Returns:
        The extracted IR.

    Raises:
        ExportError: If model export fails.
        ConversionError: If IR conversion fails (in strict mode).

    Example:
        ```python
        import torch
        from torch_ir import extract_ir

        with torch.device('meta'):
            model = torch.nn.Linear(10, 5)
        inputs = (torch.randn(1, 10, device='meta'),)
        ir = extract_ir(model, inputs)
        print(f"Extracted {len(ir.nodes)} nodes")
        ```
    """
    # Export model using torch.export
    exported = export_model(model, example_inputs, strict=strict)

    # Get model name
    if model_name is None:
        model_name = get_model_name(model)

    # Convert to IR
    ir = convert_exported_program(exported, model_name=model_name, strict=False)

    # Capture lifted tensor constants from export
    if hasattr(exported, "constants") and exported.constants:
        for name, tensor in exported.constants.items():
            if tensor.device.type == "meta":
                raise ConversionError(
                    f"Lifted tensor constant '{name}' is on meta device and has no data. "
                    f"This happens when forward() creates tensors via torch.tensor(...). "
                    f"Fix: use self.register_buffer('{name}', tensor) in __init__() instead, "
                    f"or pass a CPU-device model to extract_ir()."
                )
        ir.constants = dict(exported.constants)

    return ir
