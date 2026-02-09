"""Verifier for comparing original model output vs IR execution output."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .executor import execute_ir
from .ir import IR
from .weight_loader import load_weights


@dataclass
class VerificationReport:
    """Report from verification comparison.

    Attributes:
        is_valid: Whether all outputs are within tolerance.
        max_diff: Maximum absolute difference across all output tensors.
        mean_diff: Maximum of per-output mean absolute differences.
        num_outputs: Number of output tensors compared.
        output_details: Per-output comparison details (index, shape, is_close, max_diff, mean_diff).
        error_message: Human-readable error description when verification fails. ``None`` on success.
    """

    is_valid: bool
    max_diff: float = 0.0
    mean_diff: float = 0.0
    num_outputs: int = 0
    output_details: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None

    def __str__(self) -> str:
        if self.is_valid:
            return (
                f"Verification PASSED\n"
                f"  Outputs compared: {self.num_outputs}\n"
                f"  Max difference: {self.max_diff:.2e}\n"
                f"  Mean difference: {self.mean_diff:.2e}"
            )
        else:
            return (
                f"Verification FAILED\n"
                f"  Error: {self.error_message}\n"
                f"  Max difference: {self.max_diff:.2e}\n"
                f"  Mean difference: {self.mean_diff:.2e}"
            )


def _compare_tensors(
    original: torch.Tensor,
    ir_output: torch.Tensor,
    rtol: float,
    atol: float,
) -> Tuple[bool, float, float]:
    """Compare two tensors.

    Returns:
        Tuple of (is_close, max_diff, mean_diff)
    """
    if original.shape != ir_output.shape:
        return False, float("inf"), float("inf")

    if original.numel() == 0:
        return True, 0.0, 0.0

    # Convert to same dtype for comparison
    original = original.float()
    ir_output = ir_output.float()

    diff = torch.abs(original - ir_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    is_close = torch.allclose(original, ir_output, rtol=rtol, atol=atol)

    return is_close, max_diff, mean_diff


def _verify_outputs(
    original_outputs: Tuple[torch.Tensor, ...],
    ir_outputs: Tuple[torch.Tensor, ...],
    rtol: float,
    atol: float,
) -> Tuple[bool, VerificationReport]:
    """Compare original and IR outputs and produce a verification report."""
    if len(original_outputs) != len(ir_outputs):
        return False, VerificationReport(
            is_valid=False,
            num_outputs=len(ir_outputs),
            error_message=(f"Output count mismatch: original={len(original_outputs)}, ir={len(ir_outputs)}"),
        )

    all_close = True
    max_diff_overall = 0.0
    mean_diff_overall = 0.0
    output_details = []

    for i, (orig, ir_out) in enumerate(zip(original_outputs, ir_outputs)):
        is_close, max_diff, mean_diff = _compare_tensors(orig, ir_out, rtol, atol)

        output_details.append(
            {
                "index": i,
                "shape": list(orig.shape),
                "is_close": is_close,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
            }
        )

        if not is_close:
            all_close = False

        max_diff_overall = max(max_diff_overall, max_diff)
        mean_diff_overall = max(mean_diff_overall, mean_diff)

    report = VerificationReport(
        is_valid=all_close,
        max_diff=max_diff_overall,
        mean_diff=mean_diff_overall,
        num_outputs=len(ir_outputs),
        output_details=output_details,
        error_message=None if all_close else "Output tensors do not match within tolerance",
    )

    return all_close, report


def _run_original_model(
    original_model: nn.Module,
    test_inputs: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, ...]:
    """Run the original model and normalize outputs to a tuple."""
    original_model.eval()
    with torch.no_grad():
        original_outputs = original_model(*test_inputs)

    if isinstance(original_outputs, torch.Tensor):
        original_outputs = (original_outputs,)
    elif not isinstance(original_outputs, tuple):
        original_outputs = tuple(original_outputs)

    return original_outputs


def verify_ir(
    ir: IR,
    weights_path: Union[str, Path],
    original_model: nn.Module,
    test_inputs: Tuple[torch.Tensor, ...],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Tuple[bool, VerificationReport]:
    """Verify that IR execution matches original model output.

    Args:
        ir: The IR graph to verify.
        weights_path: Path to the weight file.
        original_model: The original PyTorch model (with weights loaded).
        test_inputs: Test input tensors.
        rtol: Relative tolerance for torch.allclose.
        atol: Absolute tolerance for torch.allclose.

    Returns:
        Tuple of (is_valid, report).
    """
    try:
        original_outputs = _run_original_model(original_model, test_inputs)
        weights = load_weights(weights_path)
        ir_outputs = execute_ir(ir, test_inputs, weights=weights)
        return _verify_outputs(original_outputs, ir_outputs, rtol, atol)
    except Exception as e:
        return False, VerificationReport(
            is_valid=False,
            error_message=f"Verification error: {str(e)}",
        )


def verify_ir_with_state_dict(
    ir: IR,
    state_dict: Dict[str, torch.Tensor],
    original_model: nn.Module,
    test_inputs: Tuple[torch.Tensor, ...],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Tuple[bool, VerificationReport]:
    """Verify IR execution using a state dict instead of file.

    Args:
        ir: The IR graph to verify.
        state_dict: The weight state dict.
        original_model: The original PyTorch model (with weights loaded).
        test_inputs: Test input tensors.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        Tuple of (is_valid, report).
    """
    try:
        original_outputs = _run_original_model(original_model, test_inputs)
        ir_outputs = execute_ir(ir, test_inputs, weights=state_dict)
        return _verify_outputs(original_outputs, ir_outputs, rtol, atol)
    except Exception as e:
        return False, VerificationReport(
            is_valid=False,
            error_message=f"Verification error: {str(e)}",
        )
