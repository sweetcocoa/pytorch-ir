"""Weight loader for .pt and .safetensors files."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from .ir import NPU_IR


class WeightLoadError(Exception):
    """Raised when weight loading fails."""

    pass


def _dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    return dtype_map.get(dtype_str, torch.float32)


def load_weights_pt(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Load weights from a .pt file.

    Args:
        path: Path to the .pt file.

    Returns:
        Dictionary mapping parameter names to tensors.

    Raises:
        WeightLoadError: If loading fails.
    """
    path = Path(path)

    if not path.exists():
        raise WeightLoadError(f"Weight file not found: {path}")

    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        return state_dict
    except Exception as e:
        raise WeightLoadError(f"Failed to load weights from {path}: {e}") from e


def load_weights_safetensors(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Load weights from a .safetensors file.

    Args:
        path: Path to the .safetensors file.

    Returns:
        Dictionary mapping parameter names to tensors.

    Raises:
        WeightLoadError: If loading fails.
        ImportError: If safetensors is not installed.
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError(
            "safetensors package is required for loading .safetensors files.\nInstall it with: pip install safetensors"
        )

    path = Path(path)

    if not path.exists():
        raise WeightLoadError(f"Weight file not found: {path}")

    try:
        return load_file(path)
    except Exception as e:
        raise WeightLoadError(f"Failed to load weights from {path}: {e}") from e


def load_weights(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Load weights from a file (auto-detect format).

    Args:
        path: Path to the weight file (.pt or .safetensors).

    Returns:
        Dictionary mapping parameter names to tensors.

    Raises:
        WeightLoadError: If loading fails or format is unknown.
    """
    path = Path(path)

    if path.suffix == ".safetensors":
        return load_weights_safetensors(path)
    elif path.suffix in (".pt", ".pth", ".bin"):
        return load_weights_pt(path)
    else:
        # Try .pt format as default
        try:
            return load_weights_pt(path)
        except Exception:
            raise WeightLoadError(
                f"Unknown weight file format: {path.suffix}\nSupported formats: .pt, .pth, .bin, .safetensors"
            )


def validate_weights_against_ir(
    weights: Dict[str, torch.Tensor],
    ir: NPU_IR,
) -> List[str]:
    """Validate that loaded weights match the IR weight metadata.

    Checks for missing weights and shape/dtype mismatches between the loaded
    tensors and the metadata recorded during IR extraction.

    Args:
        weights: Loaded weight dictionary (``state_dict``).
        ir: The IR containing weight metadata.

    Returns:
        List of human-readable validation error strings. Empty when all weights match.
    """
    errors = []

    ir_weight_names = {w.name for w in ir.weights}
    loaded_weight_names = set(weights.keys())

    # Check for missing weights
    missing = ir_weight_names - loaded_weight_names
    if missing:
        errors.append(f"Missing weights: {missing}")

    # Check for shape/dtype mismatches
    for weight_meta in ir.weights:
        if weight_meta.name not in weights:
            continue

        loaded_tensor = weights[weight_meta.name]
        expected_shape = tuple(weight_meta.shape)
        actual_shape = tuple(loaded_tensor.shape)

        if expected_shape != actual_shape:
            errors.append(f"Shape mismatch for '{weight_meta.name}': expected {expected_shape}, got {actual_shape}")

        expected_dtype = _dtype_str_to_torch(weight_meta.dtype)
        if loaded_tensor.dtype != expected_dtype:
            errors.append(
                f"Dtype mismatch for '{weight_meta.name}': expected {expected_dtype}, got {loaded_tensor.dtype}"
            )

    return errors


class WeightLoader:
    """Class-based interface for weight loading."""

    def __init__(self, validate: bool = True):
        """Initialize the loader.

        Args:
            validate: If True, validate weights against IR metadata.
        """
        self.validate = validate

    def load(
        self,
        path: Union[str, Path],
        ir: Optional[NPU_IR] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load weights from a file.

        Args:
            path: Path to the weight file.
            ir: Optional IR for validation.

        Returns:
            Dictionary mapping parameter names to tensors.

        Raises:
            WeightLoadError: If loading or validation fails.
        """
        weights = load_weights(path)

        if self.validate and ir is not None:
            errors = validate_weights_against_ir(weights, ir)
            if errors:
                raise WeightLoadError("Weight validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return weights

    def load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        ir: Optional[NPU_IR] = None,
    ) -> Dict[str, torch.Tensor]:
        """Use an existing state dict as weights.

        Args:
            state_dict: The state dict to use.
            ir: Optional IR for validation.

        Returns:
            The same state dict (after validation if enabled).

        Raises:
            WeightLoadError: If validation fails.
        """
        if self.validate and ir is not None:
            errors = validate_weights_against_ir(state_dict, ir)
            if errors:
                raise WeightLoadError("Weight validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return state_dict
