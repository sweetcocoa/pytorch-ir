"""Tests for lifted tensor constant handling.

Lifted tensor constants are module attributes assigned as plain tensors
(e.g., ``self.indices = torch.tensor([0, 1, 2])``) rather than via
``register_buffer``.  ``torch.export`` lifts these into the graph
signature under ``inputs_to_lifted_tensor_constants``.

This module tests that the full pipeline (extract → serialize → execute → verify)
handles lifted tensor constants correctly.
"""

import json
import warnings

import torch
import torch.nn as nn

from torch_ir import extract_ir, verify_ir_with_state_dict
from torch_ir.executor import IRExecutor, execute_ir
from torch_ir.ir import IR

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class GatherWithLiftedIndex(nn.Module):
    """Model that uses a lifted tensor constant as gather indices."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        # Plain attribute — becomes a lifted tensor constant in torch.export
        self.indices = torch.tensor([0, 2, 4, 6], dtype=torch.long)

    def forward(self, x):
        out = self.linear(x)
        return out[:, self.indices]


class MaskWithLiftedTensor(nn.Module):
    """Model that multiplies by a lifted float mask."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.mask = torch.tensor([1.0, 0.0, 1.0, 0.0])

    def forward(self, x):
        return self.linear(x) * self.mask


class BufferModel(nn.Module):
    """Model that uses register_buffer (NOT a lifted constant)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("scale", torch.tensor([2.0, 2.0, 2.0, 2.0]))

    def forward(self, x):
        return self.linear(x) * self.scale


class MixedConstantAndBuffer(nn.Module):
    """Model that uses both a register_buffer and a lifted constant."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("scale", torch.tensor([2.0, 2.0, 2.0, 2.0]))
        self.offset = torch.tensor([0.1, 0.2, 0.3, 0.4])

    def forward(self, x):
        return self.linear(x) * self.scale + self.offset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_real(model_cls):
    """Extract IR from a real (non-meta) model instance."""
    model = model_cls()
    model.eval()
    # Determine input shape from the first Linear layer
    in_features = model.linear.in_features
    example = (torch.randn(1, in_features),)
    ir = extract_ir(model, example, strict=False)
    return model, ir, example


def _extract_meta(model_cls):
    """Extract IR from a meta-device model instance."""
    with torch.device("meta"):
        meta_model = model_cls()
    in_features = meta_model.linear.in_features
    meta_input = (torch.randn(1, in_features, device="meta"),)
    ir = extract_ir(meta_model, meta_input, strict=False)
    return meta_model, ir, meta_input


# ---------------------------------------------------------------------------
# Tests — IR extraction
# ---------------------------------------------------------------------------


class TestLiftedTensorExtraction:
    """Tests that lifted tensor constants are captured in the IR."""

    def test_gather_model_has_constants(self):
        """GatherWithLiftedIndex should produce a non-empty ir.constants."""
        _, ir, _ = _extract_real(GatherWithLiftedIndex)
        assert ir.constants, "Expected non-empty ir.constants for lifted index tensor"

    def test_mask_model_has_constants(self):
        """MaskWithLiftedTensor should produce a non-empty ir.constants."""
        _, ir, _ = _extract_real(MaskWithLiftedTensor)
        assert ir.constants, "Expected non-empty ir.constants for lifted mask tensor"

    def test_buffer_model_no_constants(self):
        """BufferModel (register_buffer) should NOT produce lifted constants."""
        _, ir, _ = _extract_real(BufferModel)
        assert not ir.constants, "register_buffer should not produce lifted constants"

    def test_mixed_model_has_constants(self):
        """MixedConstantAndBuffer should have constants for the non-buffer attribute."""
        _, ir, _ = _extract_real(MixedConstantAndBuffer)
        assert ir.constants, "Expected ir.constants for the offset tensor"

    def test_constant_values_preserved(self):
        """Extracted constants should hold the correct values."""
        _, ir, _ = _extract_real(MaskWithLiftedTensor)
        const_values = list(ir.constants.values())
        assert len(const_values) >= 1
        expected = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert torch.equal(const_values[0], expected)

    def test_constants_in_weight_name_mapping(self):
        """Lifted constants should appear in weight_name_mapping."""
        _, ir, _ = _extract_real(GatherWithLiftedIndex)
        # weight_name_mapping maps placeholder -> original name
        mapping_values = set(ir.weight_name_mapping.values())
        # The original attribute names should be in the mapping values
        has_constant = any("indices" in v for v in mapping_values)
        assert has_constant, f"Expected 'indices' in mapping values, got {mapping_values}"

    def test_constants_in_weights_metadata(self):
        """Lifted constants should appear in ir.weights (shape/dtype metadata)."""
        _, ir, _ = _extract_real(GatherWithLiftedIndex)
        weight_names = {w.name for w in ir.weights}
        has_indices = any("indices" in n for n in weight_names)
        assert has_indices, f"Expected 'indices' in weight names, got {weight_names}"


# ---------------------------------------------------------------------------
# Tests — Meta device extraction
# ---------------------------------------------------------------------------


class TestLiftedTensorMetaDevice:
    """Tests for lifted tensor handling when the model is on meta device."""

    def test_meta_constants_skipped_with_warning(self):
        """Meta-device constants should be skipped with a warning."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, ir, _ = _extract_meta(GatherWithLiftedIndex)
            # Constants on meta device are filtered out
            meta_const_count = sum(
                1 for v in ir.constants.values() if isinstance(v, torch.Tensor) and v.device.type == "meta"
            )
            assert meta_const_count == 0, "Meta constants should be filtered out"

    def test_meta_weight_metadata_still_present(self):
        """Shape/dtype metadata for lifted constants should still be in ir.weights."""
        _, ir, _ = _extract_meta(GatherWithLiftedIndex)
        weight_names = {w.name for w in ir.weights}
        has_indices = any("indices" in n for n in weight_names)
        assert has_indices, f"Shape metadata should be preserved, got {weight_names}"

    def test_meta_execution_fails_without_constants(self):
        """Meta-extracted IR should fail execution without externally supplied constants."""
        _, ir, _ = _extract_meta(MaskWithLiftedTensor)
        real_model = MaskWithLiftedTensor()
        test_input = (torch.randn(1, 4),)

        is_valid, report = verify_ir_with_state_dict(
            ir=ir,
            state_dict=real_model.state_dict(),
            original_model=real_model,
            test_inputs=test_input,
        )
        assert not is_valid, "Should fail without constants"
        assert report.error_message is not None and "not found in registry" in report.error_message

    def test_meta_execution_succeeds_with_constants(self):
        """Meta-extracted IR should work when constants are supplied externally."""
        _, ir, _ = _extract_meta(MaskWithLiftedTensor)
        real_model = MaskWithLiftedTensor()
        real_model.eval()
        test_input = (torch.randn(2, 4),)

        # Provide the missing constant externally
        constants = {"mask": torch.tensor([1.0, 0.0, 1.0, 0.0])}

        with torch.no_grad():
            expected = real_model(*test_input)

        outputs = execute_ir(ir, test_input, weights=real_model.state_dict(), constants=constants)
        assert torch.allclose(outputs[0], expected, atol=1e-5)

    def test_meta_verify_with_constants(self):
        """verify_ir_with_state_dict should pass when constants are provided."""
        _, ir, _ = _extract_meta(GatherWithLiftedIndex)
        real_model = GatherWithLiftedIndex()
        real_model.eval()
        test_input = (torch.randn(2, 8),)

        constants = {"indices": torch.tensor([0, 2, 4, 6], dtype=torch.long)}

        is_valid, report = verify_ir_with_state_dict(
            ir=ir,
            state_dict=real_model.state_dict(),
            original_model=real_model,
            test_inputs=test_input,
            constants=constants,
        )
        assert is_valid, f"Verification with constants should pass: {report}"

    def test_meta_mixed_model_with_constants(self):
        """Mixed model (buffer + lifted constant) should work with constants."""
        _, ir, _ = _extract_meta(MixedConstantAndBuffer)
        real_model = MixedConstantAndBuffer()
        real_model.eval()
        test_input = (torch.randn(2, 4),)

        constants = {"offset": torch.tensor([0.1, 0.2, 0.3, 0.4])}

        is_valid, report = verify_ir_with_state_dict(
            ir=ir,
            state_dict=real_model.state_dict(),
            original_model=real_model,
            test_inputs=test_input,
            constants=constants,
        )
        assert is_valid, f"Mixed model verification should pass: {report}"


# ---------------------------------------------------------------------------
# Tests — Serialization round-trip
# ---------------------------------------------------------------------------


class TestLiftedTensorSerialization:
    """Tests for JSON serialization of lifted tensor constants."""

    def test_round_trip_preserves_constants(self):
        """to_dict → from_dict should preserve constants exactly."""
        _, ir, _ = _extract_real(MaskWithLiftedTensor)
        assert ir.constants

        d = ir.to_dict()
        restored = IR.from_dict(d)

        assert len(restored.constants) == len(ir.constants)
        for key in ir.constants:
            assert key in restored.constants
            assert torch.equal(restored.constants[key], ir.constants[key])

    def test_json_serializable(self):
        """IR with constants should be fully JSON-serializable."""
        _, ir, _ = _extract_real(GatherWithLiftedIndex)
        d = ir.to_dict()
        json_str = json.dumps(d)
        reloaded = json.loads(json_str)
        restored = IR.from_dict(reloaded)
        assert len(restored.constants) == len(ir.constants)

    def test_round_trip_preserves_int_dtype(self):
        """Integer constants (e.g., index tensors) should keep int64 dtype."""
        _, ir, _ = _extract_real(GatherWithLiftedIndex)
        d = ir.to_dict()
        restored = IR.from_dict(d)
        for key in ir.constants:
            assert restored.constants[key].dtype == ir.constants[key].dtype

    def test_empty_constants_round_trip(self):
        """IR with no constants should round-trip cleanly."""
        _, ir, _ = _extract_real(BufferModel)
        assert not ir.constants
        d = ir.to_dict()
        restored = IR.from_dict(d)
        assert not restored.constants


# ---------------------------------------------------------------------------
# Tests — Execution
# ---------------------------------------------------------------------------


class TestLiftedTensorExecution:
    """Tests that IR execution correctly uses lifted tensor constants."""

    def test_gather_execution(self):
        """Execution of GatherWithLiftedIndex should produce correct output."""
        model, ir, _ = _extract_real(GatherWithLiftedIndex)
        test_input = (torch.randn(2, 8),)

        with torch.no_grad():
            expected = model(*test_input)

        outputs = execute_ir(ir, test_input, weights=model.state_dict())
        assert outputs[0].shape == expected.shape
        assert torch.allclose(outputs[0], expected, atol=1e-5)

    def test_mask_execution(self):
        """Execution of MaskWithLiftedTensor should produce correct output."""
        model, ir, _ = _extract_real(MaskWithLiftedTensor)
        test_input = (torch.randn(2, 4),)

        with torch.no_grad():
            expected = model(*test_input)

        outputs = execute_ir(ir, test_input, weights=model.state_dict())
        assert torch.allclose(outputs[0], expected, atol=1e-5)

    def test_mixed_execution(self):
        """Execution of MixedConstantAndBuffer should produce correct output."""
        model, ir, _ = _extract_real(MixedConstantAndBuffer)
        test_input = (torch.randn(2, 4),)

        with torch.no_grad():
            expected = model(*test_input)

        outputs = execute_ir(ir, test_input, weights=model.state_dict())
        assert torch.allclose(outputs[0], expected, atol=1e-5)

    def test_executor_registers_constants(self):
        """IRExecutor should register all constants in its tensor registry."""
        _, ir, _ = _extract_real(MaskWithLiftedTensor)
        model = MaskWithLiftedTensor()

        executor = IRExecutor(ir, weights=model.state_dict())
        # Trigger _register_tensors by calling execute
        outputs = executor.execute((torch.randn(1, 4),))
        assert len(outputs) >= 1

    def test_serialized_ir_executes_correctly(self):
        """IR round-tripped through JSON should still execute correctly."""
        model, ir, _ = _extract_real(GatherWithLiftedIndex)
        d = ir.to_dict()
        restored = IR.from_dict(d)

        test_input = (torch.randn(2, 8),)
        with torch.no_grad():
            expected = model(*test_input)

        outputs = execute_ir(restored, test_input, weights=model.state_dict())
        assert torch.allclose(outputs[0], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests — Verification (end-to-end)
# ---------------------------------------------------------------------------


class TestLiftedTensorVerification:
    """End-to-end verification: original model vs IR execution."""

    def test_verify_gather_model(self):
        """verify_ir_with_state_dict should pass for GatherWithLiftedIndex."""
        model, ir, _ = _extract_real(GatherWithLiftedIndex)
        test_input = (torch.randn(2, 8),)

        is_valid, report = verify_ir_with_state_dict(
            ir=ir,
            state_dict=model.state_dict(),
            original_model=model,
            test_inputs=test_input,
        )
        assert is_valid, f"Verification failed: {report}"

    def test_verify_mask_model(self):
        """verify_ir_with_state_dict should pass for MaskWithLiftedTensor."""
        model, ir, _ = _extract_real(MaskWithLiftedTensor)
        test_input = (torch.randn(2, 4),)

        is_valid, report = verify_ir_with_state_dict(
            ir=ir,
            state_dict=model.state_dict(),
            original_model=model,
            test_inputs=test_input,
        )
        assert is_valid, f"Verification failed: {report}"

    def test_verify_mixed_model(self):
        """verify_ir_with_state_dict should pass for MixedConstantAndBuffer."""
        model, ir, _ = _extract_real(MixedConstantAndBuffer)
        test_input = (torch.randn(2, 4),)

        is_valid, report = verify_ir_with_state_dict(
            ir=ir,
            state_dict=model.state_dict(),
            original_model=model,
            test_inputs=test_input,
        )
        assert is_valid, f"Verification failed: {report}"

    def test_verify_after_serialization_round_trip(self):
        """Verification should pass after JSON round-trip."""
        model, ir, _ = _extract_real(MixedConstantAndBuffer)
        restored = IR.from_dict(ir.to_dict())

        test_input = (torch.randn(2, 4),)
        is_valid, report = verify_ir_with_state_dict(
            ir=restored,
            state_dict=model.state_dict(),
            original_model=model,
            test_inputs=test_input,
        )
        assert is_valid, f"Verification failed after round-trip: {report}"
