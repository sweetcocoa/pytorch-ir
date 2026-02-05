"""Model exporter using torch.export."""

import torch
import torch.nn as nn
from torch.export import ExportedProgram
from typing import Tuple, Any, Optional


class ExportError(Exception):
    """Raised when model export fails."""

    pass


def is_meta_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is on meta device."""
    return tensor.device.type == "meta"


def is_meta_module(module: nn.Module) -> bool:
    """Check if a module has all parameters on meta device."""
    for param in module.parameters():
        if not is_meta_tensor(param):
            return False
    for buffer in module.buffers():
        if not is_meta_tensor(buffer):
            return False
    return True


def validate_meta_device(model: nn.Module) -> None:
    """Validate that the model is on meta device.

    Raises:
        ExportError: If the model is not on meta device.
    """
    if not is_meta_module(model):
        raise ExportError(
            "Model must be on meta device for weight-free IR extraction.\n"
            "Convert your model using one of:\n"
            "  1. model = model.to('meta')\n"
            "  2. with torch.device('meta'):\n"
            "         model = MyModel()\n"
        )


def validate_inputs_meta(inputs: Tuple[Any, ...]) -> None:
    """Validate that all tensor inputs are on meta device.

    Raises:
        ExportError: If any tensor input is not on meta device.
    """
    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor) and not is_meta_tensor(inp):
            raise ExportError(
                f"Input tensor at index {i} must be on meta device.\n"
                f"Current device: {inp.device}\n"
                "Use: torch.randn(..., device='meta')"
            )


def export_model(
    model: nn.Module,
    example_inputs: Tuple[Any, ...],
    *,
    strict: bool = True,
) -> ExportedProgram:
    """Export a model using torch.export.

    Args:
        model: The PyTorch model to export (must be on meta device).
        example_inputs: Example inputs for tracing (must be on meta device).
        strict: If True, validate meta device. Set False for testing with real tensors.

    Returns:
        ExportedProgram containing the traced graph.

    Raises:
        ExportError: If validation fails or export encounters an error.
    """
    if strict:
        validate_meta_device(model)
        validate_inputs_meta(example_inputs)

    try:
        # torch.export uses FakeTensor internally (meta tensor subclass)
        exported = torch.export.export(model, example_inputs)
        return exported
    except Exception as e:
        # Provide helpful error messages for common issues
        error_msg = str(e)

        if "dynamic control flow" in error_msg.lower():
            raise ExportError(
                f"Model contains dynamic control flow which cannot be traced.\n"
                f"Data-dependent if/for statements are not supported.\n"
                f"Original error: {e}"
            ) from e

        if "getattr" in error_msg.lower() and "none" in error_msg.lower():
            raise ExportError(
                f"Model may have uninitialized attributes or None values.\n"
                f"Ensure all model components are properly initialized.\n"
                f"Original error: {e}"
            ) from e

        raise ExportError(f"Failed to export model: {e}") from e


def get_model_name(model: nn.Module) -> str:
    """Extract model name from module class."""
    return model.__class__.__name__
