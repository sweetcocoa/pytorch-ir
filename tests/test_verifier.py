"""Tests for the IR verifier."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os

from npu_ir import extract_ir
from npu_ir.verifier import (
    verify_ir,
    verify_ir_with_state_dict,
    IRVerifier,
    VerificationReport,
    _compare_tensors,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestCompareTensors:
    """Tests for tensor comparison."""

    def test_identical_tensors(self):
        """Test comparing identical tensors."""
        t = torch.randn(2, 3)
        is_close, max_diff, mean_diff = _compare_tensors(t, t.clone(), 1e-5, 1e-5)

        assert is_close
        assert max_diff == 0.0
        assert mean_diff == 0.0

    def test_different_tensors(self):
        """Test comparing different tensors."""
        t1 = torch.zeros(2, 3)
        t2 = torch.ones(2, 3)

        is_close, max_diff, mean_diff = _compare_tensors(t1, t2, 1e-5, 1e-5)

        assert not is_close
        assert max_diff == 1.0
        assert mean_diff == 1.0

    def test_shape_mismatch(self):
        """Test comparing tensors with different shapes."""
        t1 = torch.randn(2, 3)
        t2 = torch.randn(3, 2)

        is_close, max_diff, mean_diff = _compare_tensors(t1, t2, 1e-5, 1e-5)

        assert not is_close
        assert max_diff == float("inf")


class TestVerificationReport:
    """Tests for VerificationReport."""

    def test_report_str_passed(self):
        """Test string representation for passed report."""
        report = VerificationReport(
            is_valid=True,
            max_diff=1e-7,
            mean_diff=1e-8,
            num_outputs=1,
        )

        report_str = str(report)
        assert "PASSED" in report_str
        assert "1e-07" in report_str or "1.00e-07" in report_str

    def test_report_str_failed(self):
        """Test string representation for failed report."""
        report = VerificationReport(
            is_valid=False,
            max_diff=0.1,
            mean_diff=0.05,
            num_outputs=1,
            error_message="Test error",
        )

        report_str = str(report)
        assert "FAILED" in report_str
        assert "Test error" in report_str


class TestVerifyIRWithStateDict:
    """Tests for verify_ir_with_state_dict."""

    @pytest.fixture
    def simple_model_and_ir(self):
        """Create a simple model and its IR."""
        model = SimpleModel()
        model.eval()

        with torch.device("meta"):
            meta_model = SimpleModel()
        inputs = (torch.randn(1, 10, device="meta"),)
        ir = extract_ir(meta_model, inputs, strict=False)

        return model, ir

    def test_verification_passes(self, simple_model_and_ir):
        """Test that verification passes for correct IR."""
        model, ir = simple_model_and_ir
        test_inputs = (torch.randn(1, 10),)

        is_valid, report = verify_ir_with_state_dict(
            ir=ir,
            state_dict=model.state_dict(),
            original_model=model,
            test_inputs=test_inputs,
        )

        assert is_valid
        assert report.is_valid
        assert report.max_diff < 1e-5

    def test_verification_with_tolerance(self, simple_model_and_ir):
        """Test verification with different tolerances."""
        model, ir = simple_model_and_ir
        test_inputs = (torch.randn(1, 10),)

        # Tight tolerance
        is_valid, report = verify_ir_with_state_dict(
            ir=ir,
            state_dict=model.state_dict(),
            original_model=model,
            test_inputs=test_inputs,
            rtol=1e-10,
            atol=1e-10,
        )

        # Should still pass for exact computation
        assert is_valid or report.max_diff < 1e-5


class TestVerifyIR:
    """Tests for verify_ir with file."""

    def test_verification_with_file(self):
        """Test verification with saved weights file."""
        model = SimpleModel()
        model.eval()

        with torch.device("meta"):
            meta_model = SimpleModel()
        inputs = (torch.randn(1, 10, device="meta"),)
        ir = extract_ir(meta_model, inputs, strict=False)

        # Save weights to temp file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            weights_path = f.name

        try:
            test_inputs = (torch.randn(1, 10),)
            is_valid, report = verify_ir(
                ir=ir,
                weights_path=weights_path,
                original_model=model,
                test_inputs=test_inputs,
            )

            assert is_valid
        finally:
            os.unlink(weights_path)


class TestIRVerifier:
    """Tests for IRVerifier class."""

    def test_verifier_instance(self):
        """Test creating verifier instance."""
        verifier = IRVerifier(rtol=1e-4, atol=1e-4)
        assert verifier.rtol == 1e-4
        assert verifier.atol == 1e-4

    def test_verifier_verify_with_state_dict(self):
        """Test verifier.verify_with_state_dict()."""
        model = SimpleModel()
        model.eval()

        with torch.device("meta"):
            meta_model = SimpleModel()
        inputs = (torch.randn(1, 10, device="meta"),)
        ir = extract_ir(meta_model, inputs, strict=False)

        verifier = IRVerifier()
        is_valid, report = verifier.verify_with_state_dict(
            ir=ir,
            state_dict=model.state_dict(),
            original_model=model,
            test_inputs=(torch.randn(1, 10),),
        )

        assert is_valid
